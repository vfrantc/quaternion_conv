import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


if __name__ == '__main__':
    input = np.array([[[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.45, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]]])
    kernel = np.array([[[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]],
             [[0.55, 0.5, 0.55], [0.6, 0.5, 0.6], [1.0, 0.5, 0.5]]])

    print('Input: ', input)
    print('-------------------------------------')
    print('Input kernel: ', kernel)
    print('-------------------------------------')
    point_wise_mm = input*kernel
    print('After point_wise multiplication', point_wise_mm)
    ssum = point_wise_mm.sum(axis=1).sum(axis=1)
    print('After summation', ssum)


