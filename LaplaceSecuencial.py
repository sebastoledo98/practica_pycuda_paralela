#-- coding: utf-8 --
from PIL import Image
import numpy as np
import time

# Ruta de la imagen
image_path = "D:/Universidad/p67/Computacion paralela/LaplacePythonSecuancial/images/midralord.jpg"
image = Image.open(image_path)

# Mostrar información
print(f"Formato: {image.format}")
print(f"Tamano: {image.size}")
print(f"Modo: {image.mode}")
image.show()

# Convertir a escala de grises
gray = image.convert("L")
gray_array = np.array(gray, dtype=np.float32)
gray.save("D:/Universidad/p67/Computacion paralela/LaplacePythonSecuancial/images/midralord_gray.jpg")

# Función para generar kernel Laplaciano
def generate_laplacian_kernel(size):
    kernel = np.full((size, size), -1, dtype=np.float32)
    center = size // 2
    kernel[center, center] = size * size - 1
    return kernel

# Convolución manual
def manual_convolve(image, kernel):
    ksize = kernel.shape[0]
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y+ksize, x:x+ksize]
            output[y, x] = np.sum(region * kernel)
    return output

# Tamaños de kernel
kernel_sizes = [9, 21, 65, 123]

# Aplicar cada kernel
for ksize in kernel_sizes:
    print(f"\nAplicando Laplaciano {ksize}x{ksize}...")
    kernel = generate_laplacian_kernel(ksize)
    start_time = time.time()
    result = manual_convolve(gray_array, kernel)
    elapsed = time.time() - start_time
    print(f"Tiempo: {elapsed:.4f} segundos")
    result_image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    output_path = f"D:/Universidad/p67/Computacion paralela/LaplacePythonSecuancial/images/midralord_laplace_manual_{ksize}.jpg"
    result_image.save(output_path)
    result_image.show()