# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Cargar imagen y convertir a escala de grises
image_path = "D:/Universidad/p67/Computacion paralela/LaplacePythonSecuancial/images/midralord.jpg"
gray = Image.open(image_path).convert("L")
gray_array = np.array(gray).astype(np.float32)
height, width = gray_array.shape


def generate_sharpen_kernel(size, alpha=1.0):
    if size % 2 == 0 or size < 1:
        raise ValueError("El tamano del kernel debe ser un entero impar positivo")
    avg = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
    kernel = -alpha * avg
    center = size // 2
    kernel[center, center] += 1.0 + alpha
    return kernel.astype(np.float32)

# CUDA kernel para convolución manual
cuda_code = """
__global__ void sharpen_filter(float *input, float *output, float *kernel, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;

    if (x >= width || y >= height) return;

    float sum = 0.0;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            float kval = kernel[(ky + half) * ksize + (kx + half)];
            sum += input[iy * width + ix] * kval;
        }
    }
    output[y * width + x] = sum;
}
"""

mod = SourceModule(cuda_code)
sharpen_filter = mod.get_function("sharpen_filter")

# Configuración de bloques e hilos
block_dim = (16, 16, 1)
grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1])

# Tamaños de kernel
kernel_sizes = [9, 21, 65, 123]

# Aplicar cada kernel
for ksize in kernel_sizes:
    print(f"\nAplicando Sharpen {ksize}x{ksize} con bloques {block_dim}...")
    kernel = generate_sharpen_kernel(ksize)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    cuda.memcpy_htod(kernel_gpu, kernel)

    input_gpu = cuda.mem_alloc(gray_array.nbytes)
    output_gpu = cuda.mem_alloc(gray_array.nbytes)
    cuda.memcpy_htod(input_gpu, gray_array)

    start_time = time.time()
    sharpen_filter(input_gpu, output_gpu, kernel_gpu,
                   np.int32(width), np.int32(height), np.int32(ksize),
                   block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()
    elapsed = time.time() - start_time
    print(f"Tiempo: {elapsed:.4f} segundos")

    result_array = np.empty_like(gray_array)
    cuda.memcpy_dtoh(result_array, output_gpu)
    result_image = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    result_image.save(f"D:/Universidad/p67/Computacion paralela/LaplacePythonSecuancial/images/midralord_sharpen_cuda_{ksize}.jpg")
    result_image.show()