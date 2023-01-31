import numpy as np


IX, IY = np.meshgrid(np.arange(0, 96, 1.0), np.arange(0, 96, 1.0))
IX = IX / 96
IY = IY / 96


def step(p, xs):
    return 1 / ((np.exp(500 * xs - 500 * p) + 1) * (np.exp(-500 * xs - 500 * p) + 1))


def plateu(center, width, xs):
    right = center + width / 2
    left = center - width / 2
    return step(right, xs) - step(left, xs)


def point(centerx, centery, widthx, widthy, density):
    return plateu(centerx, widthx, IX) * plateu(centery, widthy, IY) * density


if __name__ == '__main__':
    import cv2

    img = point(0.7, 0.3, 0.01, 0.07, 1.0)
    cv2.imwrite('test.png', (1 - img) * 255)