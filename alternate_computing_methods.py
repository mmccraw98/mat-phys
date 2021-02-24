from numba import jit, cuda
import numpy as np
from helperfunctions import tic, toc

def cpu_func(a):
    for i in range(10000000):
        a[i] += 1

@cuda.jit(target='cuda')
def gpu_func(a):
    for i in range(10000000):
        a[i] += 1

if __name__ == '__main__':
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float32)

    tic()
    cpu_func(a)
    print("without GPU:", toc(True))

    tic()
    gpu_func(a)
    print("with GPU:", toc())