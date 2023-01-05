import cv2
import numpy as np
import matplotlib.pyplot as plt

# http://sepwww.stanford.edu/sep/jeff/Quaternions.pdf

if __name__ == '__main__':
    # read the image
    eps = np.finfo(np.float32).eps
    image = cv2.imread('lena.png')
    image = image.astype(np.float32) / 255
    h, w = image.shape[:2]

    # convert image to quaternion
    q1 = image[:, :, 0]
    q2 = image[:, :, 1]
    q3 = image[:, :, 2]
    #q0 = q1/3 + q2/3 + q3/3
    q0 = np.zeros_like(q1)

    mag = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    n_phi = 2*(q2*q3 + q0*q1)
    d_phi = q0**2 - q1**2 + q2**2 - q3**2
    n_theta = 2*(q1*q3 + q0*q2)
    d_theta = q0**2 + q1**2 - q2**2 - q3**2
    n_ksi = 2*(q1*q2 + q0*q3)

    phi = np.arctan2(n_phi, d_phi)
    theta = np.arctan2(n_theta, d_theta)
    ksi = np.arcsin(n_ksi-np.pi/4)

    plt.figure()
    plt.imshow(mag, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('mag')
    plt.savefig("figs/phase_mag.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(phi, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('phi')
    plt.savefig("figs/phase_phi.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(theta, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('theta')
    plt.savefig("figs/phase_theta.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()

    plt.figure()
    plt.imshow(ksi, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('ksi')
    plt.savefig("figs/phase_ksi.png",
                bbox_inches="tight",
                pad_inches=0,
                orientation='landscape')
    plt.show()