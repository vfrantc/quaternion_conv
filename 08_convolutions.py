import numpy as np
import matplotlib.pyplot as plt
import quaternion

def get_color(q):
    #q = np.quaternion(0.1, 0.2, 0.3, 0.4)
    r = hex(int(q.real * 255))
    x = hex(int(q.x * 255))
    y = hex(int(q.y * 255))
    z = hex(int(q.z * 255))
    return '[{} {} {} {}]'.format(r, x, y, z)

def get_colors(arr):
    h, w = arr.shape[:2]
    out = np.empty((h, w), dtype=np.object_)
    for y in range(h):
        for x in range(w):
            out[y, x] = get_color(arr[y, x])
    return out



if __name__ == '__main__':
    print('***REAL****')
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
    print('After point_wise multiplication', point_wise_mm)
    ssum = point_wise_mm.sum(axis=1).sum(axis=1)
    print('After summation', ssum)

    print('***Quaternion****')


    input = np.array([[np.quaternion(0.6, 0.6, 0.5, 0.2), np.quaternion(0.5, 0.8, 0.6, 0.2), np.quaternion(0.6, 1.0, 0.7, 0.2)],
                      [np.quaternion(0.6, 1.0, 0.6, 0.5), np.quaternion(0.5, 0.8, 0.5, 0.5), np.quaternion(0.6, 0.6, 0.6, 0.5)],
                      [np.quaternion(1.0, 0.8, 1.0, 0.8), np.quaternion(0.5, 0.9, 0.5, 0.8), np.quaternion(0.5, 0.5, 0.5, 0.8)]])

    kernel = np.array([[np.quaternion(0.5, 0, 0, 0.5), np.quaternion(0.5, 1, 1, 1), np.quaternion(0.5, 0, 0, 0.5)],
                       [np.quaternion(0.6, 0, 1, 0.6), np.quaternion(0.5, 1, 1, 0), np.quaternion(0.6, 0, 1, 0.6)],
                       [np.quaternion(1.0, 0, 0, 1.0), np.quaternion(0.5, 1, 1, 0.5), np.quaternion(0.5, 0, 0, 0.5)]])



    print('Input: ', input)
    print('-------------------------------------')
    print('Input kernel: ', kernel)
    print('-------------------------------------')
    point_wise_mm = input * kernel
    print('After point_wise multiplication', point_wise_mm)
    print('Sum of all quaternions:')
    print(sum(point_wise_mm.ravel()))

    print('\nColor input\n', get_colors(input))
    print('\nColor input\n', get_colors(kernel))






