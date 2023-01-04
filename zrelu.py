from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math


def relu(x):
	return np.maximum(0.0, x)

def f(x, y):
  c = x + 1j*y
  theta = np.arctan2(c.imag, c.real)
  indexes = np.where(np.logical_and(theta>0, theta <= np.pi/2))
  new = np.zeros_like(c)
  new[indexes] = c[indexes]
  return abs(new)

x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(16, 12))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.view_init(0, 180)
ax.set_xlabel('real')
ax.set_ylabel('imaginary')
ax.set_zlabel('absolute value')
ax.azim = -120
ax.dist = 10
ax.elev = 60
plt.title('modReLU')