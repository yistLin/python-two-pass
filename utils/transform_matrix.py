#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from copy import deepcopy
from utils.triangle import Triangle
from utils.triangle_set import TriangleSet


class TransformMatrix(object):
    """docstring for Transform Matrix"""

    def __init__(self):
        self.dim = 4
        self._matrix = np.identity(self.dim)

    def rotate(self, angle, dx=0., dy=0., dz=0.):
        angle = float(angle) * (np.pi / 180)
        vec_length = np.sqrt(dx**2 + dy**2 + dz**2)
        coord = {
            'dx': [(1, 1), (1, 2), (2, 1), (2, 2)],
            'dy': [(0, 0), (0, 2), (2, 0), (2, 2)],
            'dz': [(0, 0), (0, 1), (1, 0), (1, 1)]
        }

        vec = {
            'dx': dx,
            'dy': dy,
            'dz': dz
        }

        for key in sorted(vec.keys()):
            vec[key] *= angle / vec_length
            sin = np.sin(vec[key])
            cos = np.cos(vec[key])

            m = np.identity(self.dim)
            m[coord[key][0]] = cos
            m[coord[key][1]] = sin
            m[coord[key][2]] = -sin
            m[coord[key][3]] = cos
            self._matrix = np.dot(m, self._matrix)

    def translate(self, dx=0., dy=0., dz=0.):
        m = np.identity(self.dim)
        m[3, 0] = dx
        m[3, 1] = dy
        m[3, 2] = dz
        self._matrix = np.dot(m, self._matrix)

    def scale(self, dx=1., dy=1., dz=1.):
        m = np.identity(self.dim)
        m[0, 0] = dx
        m[1, 1] = dy
        m[2, 2] = dz
        self._matrix = np.dot(m, self._matrix)

    def shear(self, vec):
        raise NotImplementedError("Not support shear operation yet.")

    def transform(self, v):
        # v.shape: (n, 3)
        v_homo = np.ones((v.shape[0], v.shape[1] + 1))
        v_homo[:, :-1] = v
        t_v = np.dot(self._matrix.T, v_homo.T)
        factor = (1 / t_v.T[:, -1]).reshape(-1, 1)

        return t_v.T[:, :-1] / factor
