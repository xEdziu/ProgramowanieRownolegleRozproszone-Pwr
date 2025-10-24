import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import locale
locale.getpreferredencoding = lambda: "UTF-8"

kernel_code = """
__global__ void matrixMul(float *A, float *B, float *C, int N) {
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

def multiplyMatrix(N):

  A_cpu = np.random.rand(N, N).astype(np.float32)
  B_cpu = np.random.rand(N, N).astype(np.float32)
  C_cpu = np.empty((N, N), np.float32)

  A_gpu = cuda.mem_alloc(A_cpu.nbytes)
  B_gpu = cuda.mem_alloc(B_cpu.nbytes)
  C_gpu = cuda.mem_alloc(C_cpu.nbytes)

  cuda.memcpy_htod(A_gpu, A_cpu)
  cuda.memcpy_htod(B_gpu, B_cpu)

  block_size = (32, 32, 1)
  grid_size = (int(np.ceil(N/block_size[0])), int(np.ceil(N/block_size[1])), 1)


  start_time = time.time()
  matrixMul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)
  cuda.Context.synchronize()
  end_time = time.time()
  cuda.memcpy_dtoh(C_cpu, C_gpu)
  total_time = end_time - start_time
  print(f"Niskopoziomowe mnożenie macierzy na GPU trwało: {total_time:.5f} sekund dla rozmiaru {N}")

  del A_gpu
  del B_gpu
  del C_gpu

  return total_time

sizes = [256,512,1024,2048,4096]
y_cuda = []

times = []

for s in sizes:
  elapsed_time = multiplyMatrix(s)
  times.append(elapsed_time)
  acceleration = times[0]/elapsed_time
  y_cuda.append(acceleration/s)

plt.plot(sizes, y_cuda, label="Efektywność w zależności od rozmiaru macierzy")
plt.legend()
plt.xlabel("Rozmiar [liczba elementów]")
plt.ylabel("Efektywność [przyśpieszenie/liczba watkow]")
plt.show()