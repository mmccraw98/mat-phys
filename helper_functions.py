from numpy import cumsum, convolve, ones, exp, diff
from time import time
import os
import psutil

def moving_average(arr, n=10):
    ret = cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fast_moving_average(arr, n=10):
    return convolve(arr, ones(n), 'valid') / n


def integrate(arr, dX, n=10):
    return moving_average(arr, n=n)

def differentiate(arr, dX):
    return diff(arr) / dX


def tic():
    global current_time
    current_time = time()


def toc(return_numeric=False):
    if return_numeric:
        return time() - current_time
    else:
        print('{:.2f}s'.format(time() - current_time))


def get_mem_use(return_numeric=False):
    if return_numeric:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    else:
        print('{} mb used'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))


def logistic(x, a=1, s=0, k=1):
    return a / (1 + exp(-k * (x - s)))


def fft():
    pass


def ifft():
    pass


def regression():
    pass
    pass