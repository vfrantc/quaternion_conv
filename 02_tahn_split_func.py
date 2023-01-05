import math
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

eps = np.finfo(np.float32).eps


def scalar_(q):
    return q.real


def vector_(q):
    return np.quaternion(0, q.x, q.y, q.z)


def exp_(q):
    s = scalar_(q)
    v = vector_(q)
    return math.exp(s) * (math.cos(v.abs()) + (v / v.abs()) * math.sin(v.abs()))


def tanh_(q):
    return (exp_(2 * q) - 1) / (exp_(2 * q) - 1 + eps)


def construct_quaternion_array(a, b, c, d):
    if not isinstance(a, np.ndarray):
        a = np.zeros_like(b)
    if not isinstance(c, np.ndarray):
        c = np.zeros_like(b)
    if not isinstance(d, np.ndarray):
        d = np.zeros_like(b)

    h, w = b.shape[:2]
    out = np.zeros((h, w), dtype=np.quaternion)
    for y in range(h):
        for x in range(w):
            out[y, x] = np.quaternion(a[y, x], b[y, x], c[y, x], d[y, x])
    return out


def tanh(arr):
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.quaternion)
    for y in range(h):
        for x in range(w):
            out[y, x] = tanh_(arr[y, x])
    return out


def f(a, b):
    q = construct_quaternion_array(a, b, 1.0, 1.0)
    h, w = q.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            el = q[y,x]
            out[y, x] = np.quaternion(math.tanh(el.real), math.tanh(el.x), math.tanh(el.y), math.tanh(el.z)).abs()
    return out


if __name__ == '__main__':
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.view_init(0, 180)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('absolute value')
    ax.azim = -120
    ax.dist = 10
    ax.elev = 50
    plt.title('tanh split func c = 1 d = 1')

    # saving the figure.
    plt.savefig("figs/tanh_split_func.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')

    plt.show()