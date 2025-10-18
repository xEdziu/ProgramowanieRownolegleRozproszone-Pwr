import numpy as np
import cupy as cp
import time
import locale
locale.getpreferredencoding = lambda: "UTF-8"
N = 1024
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)
A_gpu = cp.array(A_cpu)
B_gpu = cp.array(B_cpu)
start_cpu = time.time()
C_cpu = np.matmul(A_cpu, B_cpu)
stop_cpu = time.time()
print(f"Mnożenie macierzy na CPU trawło: {stop_cpu - start_cpu:.5f} sekund")
start_gpu = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize() # Synchronizujemy się z GPU
stop_gpu = time.time()
print(f"Mnożenie macierzy na GPU trawło: {stop_gpu - start_gpu:.5f} sekund")