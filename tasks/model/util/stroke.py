import numpy as np
import torch as th

from torch import Tensor


def cast(element, device=-1) -> Tensor:
    element = np.array(element, dtype=np.float64)
    tensor = th.DoubleTensor(element)
    if device != -1 and th.cuda.is_available():
        return tensor.cuda(device=device)
    else:
        return tensor


IX, IY = np.meshgrid(np.arange(0, 96, 1.0), np.arange(0, 96, 1.0))
IX = cast((IX / 96).reshape(1, 1, 96, 96))
IY = cast((IY / 96).reshape(1, 1, 96, 96))


def step(p, xs):
    return 1 / ((th.exp(500 * xs - 500 * p) + 1) * (th.exp(-500 * xs - 500 * p) + 1))


def plateu(center, width, xs):
    right = center * th.ones_like(xs) + width / 2 * th.ones_like(xs)
    left = center * th.ones_like(xs) - width / 2 * th.ones_like(xs)
    return step(right, xs) - step(left, xs)


def point(centerx, centery, widthx, widthy, density):
    return plateu(centerx, widthx, IX) * plateu(centery, widthy, IY) * density


def stroke(curve, widthx, widthy, density):
    b = curve.shape[0]
    xs = curve[:, :, 0].reshape(b, -1, 1, 1)
    ys = curve[:, :, 1].reshape(b, -1, 1, 1)
    img = th.sum(point(xs, ys, widthx, widthy, density), dim=1, keepdim=True)
    return img
