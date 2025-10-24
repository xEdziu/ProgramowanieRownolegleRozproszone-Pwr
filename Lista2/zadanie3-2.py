import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import locale


y_numpy = []
y_cupy = []
times_numpy = []
times_cupy = []
sizes = [256,512,1024,2048,4096]

def multiplyMatrix(N):

  locale.getpreferredencoding = lambda: "UTF-8"
  A_cpu = np.random.rand(N, N).astype(np.float32)
  B_cpu = np.random.rand(N, N).astype(np.float32)
  A_gpu = cp.array(A_cpu)
  B_gpu = cp.array(B_cpu)
  start_numpy = time.time()
  C_cpu = np.matmul(A_cpu, B_cpu)
  stop_numpy = time.time()
  time_numpy = stop_numpy-start_numpy
  print(f"Mnożenie macierzy na CPU trwało: {time_numpy:.5f} sekund dla rozmiaru {N}")
  start_cupy = time.time()
  C_gpu = cp.matmul(A_gpu, B_gpu)
  cp.cuda.Stream.null.synchronize() # Synchronizujemy się z GPU
  stop_cupy = time.time()
  time_cupy = stop_cupy-start_cupy
  print(f"Mnożenie macierzy na GPU trwało: {time_cupy:.5f} sekund dla rozmiaru {N}")
  # czyszczenie pamięciu gpu
  del A_gpu
  del B_gpu
  del C_gpu

  return time_numpy, time_cupy

for s in sizes:
  elapsed_time_numpy, elapsed_time_cupy = multiplyMatrix(s)
  times_numpy.append(elapsed_time_numpy)
  times_cupy.append(elapsed_time_cupy)
  acceleration_numpy = times_numpy[0]/elapsed_time_numpy
  y_numpy.append(acceleration_numpy/s)
  acceleration_cupy = times_cupy[0]/elapsed_time_cupy
  y_cupy.append(acceleration_cupy/s)

plt.plot(sizes, y_numpy, label="Numpy (CPU)")
plt.plot(sizes, y_cupy, label="Cupy (GPU)")
plt.legend()
plt.title("Efektywność w zależności od rozmiaru macierzy")
plt.xlabel("Rozmiar [liczba elementów]")
plt.ylabel("Efektywność [przyśpieszenie/liczba watkow]")
plt.show()