import matplotlib.pyplot as plt
import numpy as np

plt.plot(sizes, y_numpy, 'bo--', label="Numpy - CPU")
plt.plot(sizes, y_cupy, 'go--', label="Cupy - GPU")
plt.plot(sizes, y_cuda, 'ro--', label="Cuda - GPU")
plt.legend()
plt.title("Efektywność w zależności od rozmiaru macierzy")
plt.xlabel("Rozmiar [liczba elementów]")
plt.ylabel("Efektywność [przyśpieszenie/liczba watkow]")
plt.show()