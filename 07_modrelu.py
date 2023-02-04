import cv2
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

def relu(x):
    return np.maximum(0.0, x)

'''
def f(x, y):
  c = x + 1j*y
  theta = np.arctan2(c.imag, c.real)
  indexes = np.where(np.logical_and(theta>0, theta <= np.pi/2))
  new = np.zeros_like(c)
  new[indexes] = c[indexes]
  return abs(new)
'''

def relu(x):
	return np.maximum(0.0, x)

def f(a, b):
    q = construct_quaternion_array(a, b, 1.0, 1.0)
    h, w = q.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            q0 = q[y,x].real
            q1 = q[y,x].x
            q2 = q[y,x].y
            q3 = q[y,x].z
            mag = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
            n_phi = 2 * (q2 * q3 + q0 * q1)
            d_phi = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2
            n_theta = 2 * (q1 * q3 + q0 * q2)
            d_theta = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
            n_ksi = 2 * (q1 * q2 + q0 * q3)

            phi = np.arctan2(n_phi, d_phi)
            theta = np.arctan2(n_theta, d_theta)
            ksi = np.arcsin(n_ksi)
            radius = 1.0

            res = relu(mag + radius) * np.exp(np.quaternion(0, 1, 0, 0) * phi) * np.exp(np.quaternion(0, 0, 1, 0) * theta) * np.exp(np.quaternion(0, 0, 0, 1)*ksi)
            out[y, x] = res.abs()
    return out

def imsave(fname, img):
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)

if __name__ == '__main__':
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
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
    plt.title('relu split function c = 1 d = 1')

    # saving the figure.
    plt.savefig("figs/modrelu.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')

    plt.show()

    img = cv2.imread('gc-exterior-AI-3200x1800.jpg')
    img = cv2.resize(img, (400, 300))
    img = img.astype(np.float32) / 255 - 0.5
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            q0 = 0.0
            q1 = img[y, x, 0]
            q2 = img[y, x, 1]
            q3 = img[y, x, 2]
            mag = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
            n_phi = 2 * (q2 * q3 + q0 * q1)
            d_phi = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2
            n_theta = 2 * (q1 * q3 + q0 * q2)
            d_theta = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
            n_ksi = 2 * (q1 * q2 + q0 * q3)

            phi = np.arctan2(n_phi, d_phi)
            theta = np.arctan2(n_theta, d_theta)
            ksi = np.arcsin(n_ksi)
            radius = 1.0

            res = relu(mag + radius) * np.exp(np.quaternion(0, 1, 0, 0) * phi) * np.exp(np.quaternion(0, 0, 1, 0) * theta) * np.exp(np.quaternion(0, 0, 0, 1)*ksi)
            out[y, x, 0] = res.w
            out[y, x, 1] = res.x
            out[y, x, 2] = res.y
            out[y, x, 3] = res.z

    imsave('figs/activation_modrelu_q0.png', out[:, :, 0])
    imsave('figs/activation_modrelu_q1.png', out[:, :, 1])
    imsave('figs/activation_modrelu_q2.png', out[:, :, 2])
    imsave('figs/activation_modrelu_q3.png', out[:, :, 3])
    imsave('figs/activation_modrelu.png', out[:, :, 1:])

