from numpy import cumsum, convolve, ones, exp, diff
from time import time
import psutil
import pickle
import os
import re


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
        print('process completed in {:.2f}s'.format(time() - current_time))


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


def safesave(thing, path, overwrite=False):
    '''
    safely saves a thing to a given path, avoids overwrite issues
    :param thing: obj thing to be saved
    :param path: os.path path where the thing will be saved
    :param overwrite: bool determines whether or not to overwrite
    :return: none
    '''

    def getnextfile(filename):
        '''
        given a file name, with an extension, find the next numeric instance of the file name
        i.e. the_file1.csv -> the_file2.csv
        :param file_name: str file name with a file extension
        :return: str the next numeric instance of the file name
        '''
        name, extension = filename.split(sep='.')  # split the file name at the file extension
        # \d indicates numeric digits, $ indicates the end of the string
        stripped_name = re.sub(r'\d+$', '', name)  # remove any numbers at the end of the file
        fnum = re.search(r'\d+$', name)  # get any numbers at the end of the file
        # if there are any numbers at the end of the file, add 1 to get the next file number and cast it as a string
        # if there aren't any numbers at the end of the file, use 1 as the next number
        next_fnum = str(int(fnum.group()) + 1) if fnum is not None else '1'
        return stripped_name + next_fnum + '.' + extension  # return the next file number

    # defining some variables that will be used often
    dir_name = os.path.dirname(path)  # directory name
    file_name = os.path.basename(path)  # file name

    # check if path exists and make it if it doesn't
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # if path exists and the file exists get the next available file name and adjust the path to reflect the change of name
    # if overwrite is enabled, then skip the renaming step and just overwrite using the given path
    while os.path.isfile(path := os.path.join(dir_name, file_name)) and not overwrite:
        file_name = getnextfile(file_name)

    # get the file extension and save the file accordingly
    extension = file_name.split(sep='.')[-1]
    if extension == 'csv':
        thing.to_csv(path, index=False)
    elif extension == 'xlsx':
        thing.to_xlsx(path, index=False)
    else:
        with open(path, 'wb') as f:
            pickle.dump(thing, f)

