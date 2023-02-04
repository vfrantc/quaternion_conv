import cv2
import numpy as np
import quaternion
import matplotlib.pyplot as plt

# add subtract multiply absolute conj
# relu, crelu, zrelu, modRelu
def construct_quaternion_array(a, b, c, d=None):
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

def deconstruct_quaternion_array(qarr):
    h, w = qarr.shape[:2]
    q0 = np.zeros((h, w), dtype=np.float32)
    q1 = np.zeros((h, w), dtype=np.float32)
    q2 = np.zeros((h, w), dtype=np.float32)
    q3 = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            q0[y,x] = qarr[y, x].w
            q1[y,x] = qarr[y, x].x
            q2[y,x] = qarr[y, x].y
            q3[y,x] = qarr[y, x].z
    return q0, q1, q2, q3


def imsave(fname, img):
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)

if __name__ == '__main__':
    img1 = cv2.imread('gc-exterior-AI-3200x1800.jpg')
    img1 = cv2.resize(img1, (400, 300))
    cv2.imwrite('figs/alg_img1.png', img1)

    img2 = cv2.imread('image2.png')
    img2 = cv2.resize(img2, (400, 300))
    cv2.imwrite('figs/alg_img2.png', img2)
    print(img2.shape)

    # sum of the two
    img1 = img1.astype(np.float32) / 255
    img2 = img2.astype(np.float32) / 255
    img_sum = img1 + img2
    img_sum = img_sum / img_sum.max()
    img_sum_255 = (img_sum * 255).astype(np.uint8)
    cv2.imwrite('figs/alg_sum.png', img_sum_255)

    # diff
    img1_neg = 1 - img1
    img_neg_255 = (img1_neg * 255).astype(np.uint8)
    cv2.imwrite('figs/alg_conj.png', img_neg_255)


    img3 = -img1
    imsave('figs/alg_img1_conj_q1.png', img3[:, :, 0])
    imsave('figs/alg_img1_conj_q2.png', img3[:, :, 1])
    imsave('figs/alg_img1_conj_q3.png', img3[:, :, 2])

    # abs
    image = img1.copy()
    q1 = image[:, :, 0]
    q2 = image[:, :, 1]
    q3 = image[:, :, 2]
    q0 = np.zeros_like(q1)
    mag = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    mag = mag / mag.max()
    mag = (mag * 255).astype(np.uint8)
    cv2.imwrite('figs/alg_absolute_value.png', mag)

    imsave('figs/alg_img1_q1.png', img1[:, :, 0])
    imsave('figs/alg_img1_q2.png', img1[:, :, 1])
    imsave('figs/alg_img1_q3.png', img1[:, :, 2])

    imsave('figs/alg_img2_q1.png', img2[:, :, 0])
    imsave('figs/alg_img2_q2.png', img2[:, :, 1])
    imsave('figs/alg_img2_q3.png', img2[:, :, 2])

    # multiplication
    img = img1 * img2
    imsave('figs/alg_img_q0.png', img[:, :, 0])
    imsave('figs/alg_img_q1.png', img[:, :, 1])
    imsave('figs/alg_img_q2.png', img[:, :, 2])

    # hamiltonian product
    qimg1 = construct_quaternion_array(a=None, b=img1[:, :, 0], c=img1[:, :, 1], d=img1[:, :, 2])
    qimg2 = construct_quaternion_array(a=None, b=img2[:, :, 0], c=img2[:, :, 1], d=img2[:, :, 2])
    qimg_mul = qimg1 * qimg2
    q0, q1, q2, q3 = deconstruct_quaternion_array(qimg_mul)
    mag = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    imsave('figs/alg_hamilton_magnitude.png', mag)
    imsave('figs/alg_hamilton_q0.png', q0)
    imsave('figs/alg_hamilton_q1.png', q1)
    imsave('figs/alg_hamilton_q2.png', q2)
    imsave('figs/alg_hamilton_q3.png', q3)


