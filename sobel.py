import time
import numpy as np
from PIL import Image
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


def createSobelKernels(size, sigma):
    center = size // 2

    # Derivadas aproximadas
    Kx = np.zeros((size, size), dtype=np.float32)
    Ky = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            gx = -x * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gy = -y * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            Kx[i, j] = gx
            Ky[i, j] = gy

    # Normalización
    Kx /= np.sum(np.abs(Kx))
    Ky /= np.sum(np.abs(Ky))
    return Kx, Ky


def sobelSecuencial(array, Kx, Ky):
    rows, cols = array.shape
    kSize = Kx.shape[0]
    pad = kSize // 2
    result = np.zeros_like(array, dtype=np.float32)

    for y in range(pad, rows - pad):
        for x in range(pad, cols - pad):
            gx, gy = 0.0, 0.0
            for ky in range(kSize):
                for kx in range(kSize):
                    px = array[y + ky - pad, x + kx - pad]
                    gx += Kx[ky, kx] * px
                    gy += Ky[ky, kx] * px
            mag = np.sqrt(gx**2 + gy**2)
            result[y, x] = np.clip(mag, 0, 255)

    return result.astype(np.uint8)

mod = SourceModule("""
__global__ void sobelParalelo(unsigned char* input, unsigned char* output,
                              float* kernelX, float* kernelY,
                              int width, int height, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = kSize / 2;
    float gx = 0.0f;
    float gy = 0.0f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = x + kx;
            int py = y + ky;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                unsigned char val = input[py * width + px];
                gx += val * kernelX[(ky + half) * kSize + (kx + half)];
                gy += val * kernelY[(ky + half) * kSize + (kx + half)];
            }
        }
    }

    float mag = sqrtf(gx * gx + gy * gy);
    mag = fminf(fmaxf(mag, 0.0f), 255.0f);
    output[y * width + x] = (unsigned char)mag;
}
""")

with Image.open("nebula.jpg") as image:
    image = image.convert("L")
    image.save("image_grayscale.jpg")

    array = np.array(image, dtype=np.uint8)
    sigma = 20
    sizes = [9, 21, 65, 123]

    for kSize in sizes:
        print(f"Ejecucion sobre imagen de {array.shape} con {kSize} de kernel")

        # Crear Sobel extendido
        Kx, Ky = createSobelKernels(kSize, sigma)

        # CPU
        print("CPU (secuencial):")
        inicio = time.time()
        result = sobelSecuencial(array, Kx, Ky)
        final = time.time()
        total = final - inicio
        print(f"Tiempo en secuencial: {total:.4f} segundos")

        # GPU
        print("GPU (paralela):")
        result = np.zeros_like(array)
        block = (32, 32, 1)
        grid = (
            (array.shape[1] + block[0] - 1) // block[0],
            (array.shape[0] + block[1] - 1) // block[1],
            1,
        )

        sobel_gpu = mod.get_function("sobelParalelo")
        inicio = time.time()
        sobel_gpu( drv.In(array), drv.Out(result), drv.In(Kx.astype(np.float32)), drv.In(Ky.astype(np.float32)), np.int32(array.shape[1]), np.int32(array.shape[0]), np.int32(kSize), block=block, grid=grid)
        final = time.time()
        total = final - inicio
        print(f"Tiempo en paralelo: {total:.4f} segundos")

        Image.fromarray(result).save(f"sobel_{kSize}.jpg")

    print("\nProcesamiento completo.")
