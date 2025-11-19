import time
import numpy as np
from PIL import Image
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


def createGaussianKernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size//2
    sum = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            value = np.exp(-(x*x + y*y) / (2.0 * sigma**2))
            kernel[i, j] = value
            sum += value

    kernel /= sum
    return kernel

def gaussianoSecuencial(array, kSize, kernel):
    result = np.zeros_like(array)
    pad = kSize//2
    rows, cols = array.shape

    for y in range(rows):
        for x in range(cols):
            sum = 0.0
            for ky in range(kSize):
                for kx in range(kSize):
                    px = x + kx - pad
                    py = y + ky - pad
                    px = int(max(0, min(px, cols - 1)))
                    py = int(max(0, min(py, rows - 1)))
                    sum += array[py, px] * kernel[ky, kx]

            result[y, x] = np.clip(sum, 0, 255).astype(np.uint8)

    return result

mod = SourceModule("""
    __global__ void gaussianoParalelo(unsigned char* input, unsigned char* output, float* kernel, int width, int height, int kSize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x >= width || y >= height) return;
        int centro = kSize / 2;
        float sum = 0.0f;

        for(int ky = -centro; ky <= centro; ky++) {
            for(int kx = -centro; kx <= centro; kx++) {
                int px = x + kx;
                int py = y + ky;
                if(px >= 0 && px < width && py >= 0 && py < height) {
                    float valor = kernel[(ky + centro) * kSize + (kx + centro)];
                    sum += valor * input[py * width + px];
                }
            }
        }
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[y * width + x] = (unsigned char) sum;
    }
""")


with Image.open("nebula.jpg") as image:
    image = image.convert("L")
    image.save("image_grayscale.jpg")
    sizes = [9, 21, 65, 123]
    #sizes = [9]
    sigma = 20

    array = np.array(image, dtype=np.uint8)
    for kSize in sizes:
        kernel = createGaussianKernel(kSize, sigma)

        print("Ejecucion en secuencial sobre imagen de", array.shape, "con", kSize, "de kernel y", sigma, "de sigma")
        inicio = time.time()
        result = gaussianoSecuencial(array, kSize, kernel)
        final = time.time()
        total = final - inicio
        print(f"Tiempo en secuencial: {total:.4f} segundos")

        print("Ejecucion en paralelo con", kSize, "de kernel y", sigma, "de sigma")
        result = np.zeros_like(array)
        block = (32, 32, 1)
        grid = (
            (array.shape[1] + block[0] - 1) // block[0],
            (array.shape[0] + block[1] - 1) // block[1],
            1
        )
        paralelo = mod.get_function("gaussianoParalelo")
        inicio = time.time()
        paralelo(drv.In(array), drv.Out(result), drv.In(kernel), np.int32(array.shape[1]), np.int32(array.shape[0]), np.int32(kSize), block=block, grid=grid)
        final = time.time()
        total = final - inicio
        print(f"Tiempo en paralelo: {total:.4f} segundos")

        result = Image.fromarray(np.uint8(result))
        result = result.convert('L')
        result.save(f"image_gauss_{kSize}.jpg")
