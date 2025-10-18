import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import locale
locale.getpreferredencoding = lambda: "UTF-8"

kernel_code = """
__global void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if(row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

module = SourceModule(kernel_code)
matrixMul = module.get_function("matrixMul")

N = 1024

# deklaracja zmiennych
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)
C_cpu = np.empty((N, N), np.float32)

# przypisanie pamięci
A_gpu = cuda.mem_alloc(A_cpu.nbytes)
B_gpu = cuda.mem_alloc(B_cpu.nbytes)
C_gpu = cuda.mem_alloc(C_cpu.nbytes)

# przesłanie danych na z cpu na gpu
cuda.memcpy_htod(A_gpu, A_cpu)
cuda.memcpy_htod(B_gpu, B_cpu)

# definiowanie rozmiaru bloku i siatki
block_size = (32, 32, 1)
grid_size = (int(N/32), int(N/32), 1)

# uruchomienie działania na gpu, synchronizacja i pobieranie danych
start_time = time.time()
matrixMul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)
cuda.Context.synchronize()
end_time = time.time()
cuda.memcpy_dtoh(C_cpu, C_gpu)
print(f"Niskopoziomowe mnożenie macierzy na GPU trawło: {end_time - start_time:.5f} sekund")

# zwolnienie zablokowanych zmiennych na gpu
del A_gpu
del B_gpu
del C_gpu