from pprint import pprint
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input = np.array([[[0.6, 0.5, 0.6], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
                      [[0.6, 0.8, 1.0], [1.0, 0.8, 0.6], [0.8, 0.9, 0.5]],
                      [[0.5, 0.6, 0.7], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
                      [[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]])
    kernel = np.array([[[0.5, 0.5, 0.5], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
                       [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                       [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                       [[0.5, 1, 0.5], [0.6, 0, 0.6], [1.0, 0.5, 0.5]]])

    print('Input: ', input)
    print('-------------------------------------')
    print('Input kernel: ', kernel)
    print('-------------------------------------')
    point_wise_mm = input*kernel
    print('After point_wise multiplication', point_wise_mm.ravel())
    ssum = point_wise_mm.sum(axis=1).sum(axis=1)
    print('After summation', ssum)

    print()


