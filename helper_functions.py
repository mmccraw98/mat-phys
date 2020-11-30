from numpy import cumsum, convolve, ones, exp
from time import time


def moving_average(arr, n=10):
    ret = cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fast_moving_average(arr, n=10):
    return convolve(arr, ones(n), 'valid') / n


def tic():
    global current_time
    current_time = time()


def toc(returns=False):
    if returns:
        return time() - current_time
    else:
        print('{:.2f}s'.format(time() - current_time))


def logistic(x, a=1, s=0, k=1):
    return a / (1 + exp(-k * (x - s)))


def fft():
    pass

def ifft():
    pass