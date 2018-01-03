#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from numpy.core.umath_tests import inner1d

from utils import read_tri, Triangle


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
    return np.dot(rot_mat, mat)


def _trace_ray(ray_ori, ray_drt, mat_p, mat_n, mat_c, mat_light, reflection, depth, test_hit=False):

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
        return None, None, None

    idx_min = np.argmin(dist)

    if not test_hit:
        # if hit the cover of light
        not_shadowed = []
        if np.array_equal(mat_c[idx_min, :], np.array([1., 1., 1.])):
            return mat_c[idx_min, :], dist[idx_min], pnt_int[idx_min, :]

        new_ray_ori = pnt_int[idx_min, :]
        for light_src in mat_light:
            new_ray_drt = normalize(light_src - new_ray_ori)
            _, rtn_dist, rtn_pnt = _trace_ray(
                new_ray_ori + 0.001 * new_ray_drt, new_ray_drt, mat_p, mat_n, mat_c, mat_light, reflection, 1, test_hit=True)

            if rtn_pnt is None or rtn_dist > np.linalg.norm(light_src - new_ray_ori):
                not_shadowed.append(True)
            else:
                not_shadowed.append(False)

        col_ray = np.array([0., 0., 0.])

        if any(not_shadowed):
            N = mat_n[idx_min, :]

            light_drt = mat_light[not_shadowed, :] - new_ray_ori
            light_drt = light_drt / \
                np.expand_dims(np.linalg.norm(light_drt, axis=1), axis=1)

            color = mat_c[idx_min, :]

            # ambient
            ambient = np.array([.05, .05, .05], dtype=np.float32)

            # diffuse
            diffuse = np.array(
                [1., 1., 1.]) * np.sum(np.clip(np.dot(light_drt, N), 0, np.inf), axis=0)

            # specular
            view_drt = normalize(ray_ori - new_ray_ori)
            view_ang = view_drt + light_drt
            view_ang = view_ang / \
                np.expand_dims(np.linalg.norm(view_ang, axis=1), axis=1)
            specular = np.sum(np.clip(np.dot(view_ang, N), 0,
                                      np.inf) ** 16.) * np.array([1., 1., 1.])

            col_ray = (ambient + diffuse * 0.3 + specular * 0.2) * color
    else:
        # uncover the cover of light
        dist[(mat_c == np.array([1., 1., 1.]))[:, 0]] = np.inf
        if (dist == np.inf).all():
            return None, None, None
        idx_min = np.argmin(dist)
        col_ray = mat_c[idx_min, :]

    if depth > 1:
        new_ray_drt = normalize(
            ray_drt - 2 * np.dot(ray_drt, mat_n[idx_min, :]) * mat_n[idx_min, :])
        new_ray_ori = pnt_int[idx_min, :] + 0.001 * new_ray_drt
        new_col_ray, rtn_dist, rtn_pnt = _trace_ray(
            new_ray_ori, new_ray_drt, mat_p, mat_n, mat_c, mat_light, reflection * 0.5, depth - 1)

        if new_col_ray is not None:
            col_ray = col_ray + new_col_ray * reflection

    return col_ray, dist[idx_min], pnt_int[idx_min, :]


def ray_trace(tris):
    w = 400
    h = 400
    img = np.zeros((w, h, 3))

    mat_p = np.array([tri.p for tri in tris], dtype=np.float32)
    mat_n = np.array([tri.n for tri in tris], dtype=np.float32)
    mat_c = np.array([tri.fcolor for tri in tris], dtype=np.float32)

    mat_c /= 255.

    mat_light = np.array([
        [-125., 195., -130.],
        [125., 195., -130.],
        [125., 195., -280.],
        [-125., 195., -280.]
    ])

    ori = np.array([1000., 0., 0.], dtype=np.float32)
    dst = np.array([300., 0., 0.], dtype=np.float32)

    S = (-200, -200, 200, 200)

    mat_trans = np.array([0., 250., 10.])

    ori = rotate(ori + mat_trans, -np.pi / 2, 'x')
    reflection = .25
    max_depth = 3

    ray_ori, ray_drt = [], []
    for row, y in enumerate(np.linspace(S[1], S[3], h)):
        for col, x in enumerate(np.linspace(S[0], S[2], w)):
            dst = np.array([300., x, y])
            dst = rotate(dst + mat_trans, -np.pi / 2, 'x')
            drt = normalize(dst - ori)
            ray_ori.append(ori)
            ray_drt.append(drt)

    p_trace_ray = partial(_trace_ray, mat_p=mat_p, mat_n=mat_n, mat_c=mat_c,
                          mat_light=mat_light, reflection=reflection, depth=max_depth)
    with Pool(processes=None) as pool:
        img = pool.starmap(p_trace_ray, zip(ray_ori, ray_drt))

    img = np.array(img)[:, 0]
    img = [np.zeros((3,)) if pix is None else pix for pix in img]
    img = np.clip(img, 0., 1.)
    img = img.reshape((w, h, -1))
    img = np.flipud(img)

    plt.imsave('fig.png', img)


if __name__ == '__main__':
    import sys
    tris = read_tri(sys.argv[1])
    ray_trace(tris)
