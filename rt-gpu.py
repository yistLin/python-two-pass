#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import torch
import numpy as np

from utils import xml_read_tri

try:
    from scipy.misc import imsave
except:
    from matplotlib.pyplot.plt import imsave


def inner1d(x, y):
    n, m = x.size()
    return torch.bmm(x.view(n, 1, m), y.view(n, m, 1)).squeeze(2)


def normalize(x):
    return x / np.linalg.norm(x)


def rotate(mat, rad, axis):
    rot_mats = np.array([
        [1., 0., 0.],
        [0., np.cos(rad), -np.sin(rad)],
        [0., np.sin(rad), np.cos(rad)],
        [np.cos(rad), 0., np.sin(rad)],
        [0., 1., 0.],
        [-np.sin(rad), 0., np.cos(rad)],
        [np.cos(rad), -np.sin(rad), 0.],
        [np.sin(rad), np.cos(rad), 0.],
        [0., 0., 1.]
    ])
    rot_mat = {
        'x': rot_mats[:3, :],
        'y': rot_mats[3:6, :],
        'z': rot_mats[6:, :]
    }[axis]
    return np.dot(mat, rot_mat)


class RayTracer(object):
    def __init__(self, mat_c, mat_p, mat_n, mat_e, mat_spec, mat_refl, mat_refr):
        self.num_tris = mat_p.shape[0]
        self.mat_c = np.clip(mat_c, 0., 1.)
        self.mat_p = torch.cuda.FloatTensor(mat_p)
        self.mat_n = torch.cuda.FloatTensor(mat_n)
        self.mat_e = torch.cuda.FloatTensor(mat_e)
        self.mat_spec = torch.cuda.FloatTensor(mat_spec)
        self.mat_refl = torch.cuda.FloatTensor(mat_refl)
        self.mat_refr = torch.cuda.FloatTensor(mat_refr)

        # speed up intersection test
        self.v0 = self.mat_p[:, 2] - self.mat_p[:, 0]
        self.v1 = self.mat_p[:, 1] - self.mat_p[:, 0]
        self.d00 = inner1d(self.v0, self.v0)
        self.d01 = inner1d(self.v0, self.v1)
        self.d11 = inner1d(self.v1, self.v1)
        self.invDenom = 1. / (self.d00 * self.d11 - self.d01 * self.d01)

    def trace(self, img_size, ori, dst, scene):
        img = np.zeros(img_size + (3,))
        x_coord = dst[0]
        ray_ori, ray_drt = [], []
        for row, z in enumerate(np.linspace(scene[1], scene[3], img_size[1])):
            for col, y in enumerate(np.linspace(scene[0], scene[2], img_size[0])):
                dst = np.array([x_coord, y, z])
                drt = normalize(dst - ori)
                ray_ori.append(torch.cuda.FloatTensor(dst))
                ray_drt.append(torch.cuda.FloatTensor(drt))

        img = [self._trace_ray(ori, drt) for ori, drt in zip(ray_ori, ray_drt)]

        img = np.array(img)
        img = np.clip(img, 0., 1.)
        img = img.reshape(img_size + (-1,))
        img = np.flipud(img)

        return img

    def _trace_ray(self, ray_ori, ray_drt):
        ret = self._intersect(ray_ori, ray_drt)

        if ret is None:
            return np.array([0., 0., 0.], dtype=np.float32)

        idx, pnt_int = ret

        return self.mat_c[idx, :].squeeze()

    def _intersect(self, ray_ori, ray_drt):
        ray_drt = torch.unsqueeze(ray_drt, 1)
        denom = torch.mm(self.mat_n, ray_drt) + 1e-12

        dist = inner1d(self.mat_p[:, 0, :].squeeze() -
                       ray_ori, self.mat_n) / denom

        pnt_int = dist * ray_drt.t() + ray_ori

        # Barycentric Technique
        v2 = pnt_int - self.mat_p[:, 0]
        d02 = inner1d(self.v0, v2)
        d12 = inner1d(self.v1, v2)
        u = (self.d11 * d02 - self.d01 * d12) * self.invDenom
        v = (self.d00 * d12 - self.d01 * d02) * self.invDenom

        # inside triangle
        within = (u >= 0.) & (v >= 0.) & (u + v < 1.)

        dist[~within] = np.inf
        dist[dist <= 0.] = np.inf

        if (dist == np.inf).all():
            return None

        _, idx_min = torch.min(dist, 0)

        return idx_min, pnt_int[idx_min, :]


if __name__ == '__main__':
    import sys
    mat_c, mat_p, mat_e, mat_spec, mat_refl, mat_refr = xml_read_tri(
        sys.argv[1])

    mat_p = mat_p.reshape(-1, 3)
    mat_p = rotate(mat_p, np.pi * -.42, 'x')
    mat_p = rotate(mat_p, np.pi * -.55, 'z')
    mat_p = rotate(mat_p, np.pi * -.05, 'y')
    mat_p = mat_p.reshape(-1, 3, 3)

    # normal vector
    mat_n = np.cross(mat_p[:, 0] - mat_p[:, 1], mat_p[:, 1] - mat_p[:, 2])
    mat_n = mat_n / np.expand_dims(np.linalg.norm(mat_n, axis=1), axis=1)

    tracer = RayTracer(mat_c, mat_p, mat_n, mat_e,
                       mat_spec, mat_refl, mat_refr)

    img_size = (100, 100)
    ori = np.array([40., 0., 0.], dtype=np.float32)
    dst = np.array([20., 0., 0.], dtype=np.float32)
    scene = (-15, -15, 10, 10)

    img = tracer.trace(img_size, ori, dst, scene)
    imsave('fig.png', img)

