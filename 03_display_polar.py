import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://math.stackexchange.com/questions/1496308/how-can-i-express-a-quaternion-in-polar-form

if __name__ == '__main__':
    # read the image
    eps = np.finfo(np.float32).eps
    image = cv2.imread('gc-exterior-AI-3200x1800.jpg')
    image = image.astype(np.float32) / 255
    h, w = image.shape[:2]

    # convert image to quaternion
    i = image[:, :, 0]
    j = image[:, :, 1]
    k = image[:, :, 2]
    r = i/3 + j/3 + k/3


    mag = np.sqrt(r**2 + i**2 + j**2 + k**2)
    vec_norm = np.sqrt(i**2 + j**2 + k**2)
    theta = np.arccos(r/(mag + eps))
    phi0 = i / (vec_norm + eps)
    phi1 = j / (vec_norm + eps)
    phi2 = k / (vec_norm + eps)

    plt.figure()
    plt.imshow(mag, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('mag')
    plt.savefig("figs/polar_mag.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(theta, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('theta')
    plt.savefig("figs/polar_theta.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(phi0, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('phi0')
    plt.savefig("figs/polar_phi0.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(phi1, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('phi1')
    plt.savefig("figs/polar_phi1.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(phi2, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('phi2')
    plt.savefig("figs/polar_phi2.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()
