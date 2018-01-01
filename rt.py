#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d

from utils import read_tri, Triangle

# camera view
ori = np.array([0., 0., 0.])


def normalize(x):
    return x / np.linalg.norm(x)


def _trace_ray(mat_p, mat_n, mat_c, mat_light, ray_ori, ray_drt, reflection, depth, test_hit=False):

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
        is_shadowed = False
        if np.array_equal(mat_c[idx_min, :], np.array([1., 1., 1.])):
            return mat_c[idx_min, :], dist[idx_min], pnt_int[idx_min, :]

        new_ray_ori = pnt_int[idx_min, :]
        for light_src in mat_light:
            new_ray_drt = normalize(light_src - new_ray_ori)
            _, rtn_dist, rtn_pnt = _trace_ray(
                mat_p, mat_n, mat_c, mat_light, new_ray_ori + 0.001 * new_ray_drt, new_ray_drt, reflection, 1, test_hit=True)

            if rtn_pnt is None or rtn_dist > np.linalg.norm(light_src - new_ray_ori):
                break
        else:
            is_shadowed = True

        col_ray = np.array([0., 0., 0.])

        if not is_shadowed:
            N = mat_n[idx_min, :]

            light_drt = mat_light - new_ray_ori
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
            mat_p, mat_n, mat_c, mat_light, new_ray_ori, new_ray_drt, reflection * 0.5, depth - 1)

        if new_col_ray is not None:
            col_ray = col_ray + new_col_ray * reflection

    return col_ray, dist[idx_min], pnt_int[idx_min, :]


def ray_trace(tris):
    w = 500
    h = 500
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

    # mat_p -= np.max(mat_p.reshape((-1, 3)), axis=0)

    global ori
    ori = np.array([1000., 0., 0.], dtype=np.float32)
    dst = np.array([300., 0., 0.], dtype=np.float32)

    S = (-200, -200, 200, 200)
    rad_x = -np.pi / 2
    rx = np.array([
        [1., 0., 0.],
        [0., np.cos(rad_x), -np.sin(rad_x)],
        [0., np.sin(rad_x), np.cos(rad_x)]
    ])

    rad_y = np.pi * 0.05
    ry = np.array([
        [np.cos(rad_y), 0., np.sin(rad_y)],
        [0., 1., 0.],
        [-np.sin(rad_y), 0., np.cos(rad_y)]
    ])

    rad_z = np.pi * 0.15
    rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0.],
        [np.sin(rad_z), np.cos(rad_z), 0.],
        [0., 0., 1.]
    ])

    ori = np.dot(rx, ori + np.array([0., 250., 10.]))
    for col, x in enumerate(np.linspace(S[0], S[2], w)):
        for row, y in enumerate(np.linspace(S[1], S[3], h)):
            dst = np.array([300., x, y])
            dst = np.dot(rx, dst + np.array([0., 250., 10.]))
            drt = normalize(dst - ori)
            ray_ori, ray_drt = ori, drt
            reflection = .35
            max_depth = 3

            color, _, pnt = _trace_ray(
                mat_p, mat_n, mat_c, mat_light, ray_ori, ray_drt, reflection, max_depth)
            if color is None:
                color = np.array([0., 0., 0.])

            img[h - row - 1, col, :] = np.clip(color, 0, 1)

    plt.imsave('fig.png', img)


if __name__ == '__main__':
    import sys
    tris = read_tri(sys.argv[1])
    ray_trace(tris)
