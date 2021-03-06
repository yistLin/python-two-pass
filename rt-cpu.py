#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from numpy.core.umath_tests import inner1d

from utils import XMLReader

try:
    from scipy.misc import imsave
except:
    from matplotlib.pyplot.plt import imsave


def parse_args():
    parser = argparse.ArgumentParser('Ray tracer implemented in python.')
    parser.add_argument('--input_file', type=str,
            required=True, help='input XML file')
    parser.add_argument('--output_file', type=str,
            default='fig.png', help='output figure')
    parser.add_argument('--max_depth', type=int,
            default=3, help='max depth of recursion')
    parser.add_argument('--img_size', nargs=2, type=int,
            default=(100, 100), help='resolution of output figure')

    return parser.parse_args()


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
        self.mat_c = np.clip(mat_c, 0., 1.)
        self.mat_p = mat_p
        self.mat_n = mat_n
        self.mat_e = mat_e
        self.mat_spec = mat_spec
        self.mat_refl = mat_refl
        self.mat_refr = mat_refr

        # vertex color = averaged color of neighboring triangles
        vtx_dict = {}
        mat_p_rnd = np.around(mat_p, 4)
        for idx, vtx in enumerate(mat_p_rnd.reshape(-1, 3)):
            vtx = tuple(vtx)
            if vtx not in vtx_dict:
                vtx_dict[vtx] = []

            vtx_dict[vtx].append(np.unravel_index(idx, mat_p_rnd.shape[:-1]))

        self.mat_vtx_c = np.empty(mat_p_rnd.shape, dtype=np.float32)
        for vtx, idx in vtx_dict.items():
            idx = np.array(idx)
            self.mat_vtx_c[idx[:, 0], idx[:, 1], :] = self.mat_c[idx[:, 0]].mean(0)

        # speed up intersection test
        self.v0 = self.mat_p[:, 2] - self.mat_p[:, 0]
        self.v1 = self.mat_p[:, 1] - self.mat_p[:, 0]
        self.d00 = inner1d(self.v0, self.v0)
        self.d01 = inner1d(self.v0, self.v1)
        self.d11 = inner1d(self.v1, self.v1)
        self.invDenom = 1. / (self.d00 * self.d11 - self.d01 * self.d01)

    def trace(self, img_size, ori, dst, scene, max_depth=3):
        img = np.zeros(img_size + (3,))
        x_coord = dst[0]
        ray_ori, ray_drt = [], []
        for row, z in enumerate(np.linspace(scene[1], scene[3], img_size[1])):
            for col, y in enumerate(np.linspace(scene[0], scene[2], img_size[0])):
                dst = np.array([x_coord, y, z])
                drt = normalize(dst - ori)
                ray_ori.append(dst)
                ray_drt.append(drt)

        with Pool(processes=4) as pool:
            img = pool.starmap(self._trace_ray,
                    zip(ray_ori, ray_drt, repeat(1.), repeat(max_depth)))

        img = np.array(img)
        img = np.clip(img, 0., 1.)
        img = img.reshape(img_size + (-1,))
        img = np.flipud(img)

        return img

    def _trace_ray(self, ray_ori, ray_drt, refl, depth):
        ret = self._intersect(ray_ori, ray_drt)

        if ret is None:
            return np.array([0., 0., 0.], dtype=np.float32)

        idx, pnt_int, (u, v) = ret

        # vertice color interpolation
        tri_int = self.mat_vtx_c[idx]
        color = (1. - u - v) * tri_int[0] + v * tri_int[1] + u * tri_int[2]

        refl_int = self.mat_refl[idx]

        if depth > 1 and refl * refl_int > 0.01:
            new_ray_drt = normalize(ray_drt - 2 * np.dot(ray_drt, self.mat_n[idx, :]) * self.mat_n[idx, :])
            new_ray_ori = pnt_int + 1e-3 * new_ray_drt
            ret_color = self._trace_ray(new_ray_ori, new_ray_drt, refl * refl_int, depth - 1)

            color = color + ret_color

        return refl * color

    def _intersect(self, ray_ori, ray_drt):
        denom = np.dot(self.mat_n, ray_drt) + 1e-12
        dist = inner1d(self.mat_p[:, 0, :].squeeze() -
                       ray_ori, self.mat_n) / denom

        pnt_int = ray_ori + dist.reshape((-1, 1)) * ray_drt

        # Barycentric Technique
        v2 = pnt_int - self.mat_p[:, 0]
        d02 = inner1d(self.v0, v2)
        d12 = inner1d(self.v1, v2)
        u = (self.d11 * d02 - self.d01 * d12) * self.invDenom
        v = (self.d00 * d12 - self.d01 * d02) * self.invDenom

        # inside triangle
        dist[~((u >= 0.) & (v >= 0.) & (u + v < 1.)) | (dist <= 0.)] = np.inf

        if (dist == np.inf).all():
            return None

        idx_min = np.argmin(dist)

        return idx_min, pnt_int[idx_min, :], (u[idx_min], v[idx_min])


if __name__ == '__main__':
    args = parse_args()

    mat_c, mat_p, mat_e, mat_spec, mat_refl, mat_refr = XMLReader.read_tri(
            args.input_file)

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

    img_size = tuple(args.img_size)
    ori = np.array([40., 0., 0.], dtype=np.float32)
    dst = np.array([20., 0., 0.], dtype=np.float32)
    scene = (-15, -15, 10, 10)

    img = tracer.trace(img_size, ori, dst, scene, max_depth=args.max_depth)
    imsave(args.output_file, img)

