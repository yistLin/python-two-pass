#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from numpy.core.umath_tests import inner1d

from utils import xml_read_tri


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


def _trace_ray(ray_ori, ray_drt, mat_p, mat_n, mat_c):

    denom = np.dot(mat_n, ray_drt) + 1e-12
    dist = inner1d(mat_p[:, 0, :].squeeze() - ray_ori, mat_n) / denom

    pnt_int = ray_ori + dist.reshape((-1, 1)) * ray_drt

    def same_side(d):
        p2 = mat_p[:, d[0], :].squeeze()
        a = mat_p[:, d[1], :].squeeze()
        b = mat_p[:, d[2], :].squeeze()
        cp1 = np.cross(b - a, pnt_int - a)
        cp2 = np.cross(b - a, p2 - a)
        return inner1d(cp1, cp2) >= 0

    within = np.ones((mat_p.shape[0],))
    for d in [[0, 1, 2], [1, 0, 2], [2, 0, 1]]:
        within = np.logical_and(within, same_side(d))

    dist[np.logical_not(within)] = np.inf
    dist[dist <= 0.] = np.inf

    if (dist == np.inf).all():
        return np.array([0., 0., 0.], dtype=np.float32)

    idx_min = np.argmin(dist)

    return mat_c[idx_min, :]


def ray_cast(mat_c, mat_p, mat_e, mat_spec, mat_refl, mat_refr):
    w = 100
    h = 100
    img = np.zeros((w, h, 3))

    mat_p = mat_p.reshape(-1, 3)
    mat_p = rotate(mat_p, np.pi * -.42, 'x')
    mat_p = rotate(mat_p, np.pi * -.55, 'z')
    mat_p = rotate(mat_p, np.pi * -.05, 'y')
    mat_p = mat_p.reshape(-1, 3, 3)
    print(mat_p.shape)

    # normal vector
    mat_n = np.cross(mat_p[:, 0] - mat_p[:, 1], mat_p[:, 1] - mat_p[:, 2])
    mat_n = mat_n / np.expand_dims(np.linalg.norm(mat_n, axis=1), axis=1)

    ori = np.array([40., 0., 0.], dtype=np.float32)
    dst = np.array([20., 0., 0.], dtype=np.float32)

    S = (-15, -15, 10, 10)

    mat_trans = np.array([20., 0., 0.])

    ray_ori, ray_drt = [], []
    for row, y in enumerate(np.linspace(S[1], S[3], h)):
        for col, x in enumerate(np.linspace(S[0], S[2], w)):
            dst = np.array([20., x, y])
            drt = normalize(dst - ori)
            ray_ori.append(dst)
            ray_drt.append(drt)

    p_trace_ray = partial(_trace_ray, mat_p=mat_p, mat_n=mat_n, mat_c=mat_c)
    with Pool(processes=8) as pool:
        img = pool.starmap(p_trace_ray, zip(ray_ori, ray_drt))

    img = np.array(img)
    img = np.clip(img, 0., 1.)
    img = img.reshape((w, h, -1))
    img = np.flipud(img)

    plt.imsave('fig.png', img)


if __name__ == '__main__':
    import sys
    mat_c, mat_p, mat_e, mat_spec, mat_refl, mat_refr = xml_read_tri(sys.argv[1])
    ray_cast(mat_c, mat_p, mat_e, mat_spec, mat_refl, mat_refr)

