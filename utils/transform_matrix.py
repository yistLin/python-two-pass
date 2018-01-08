#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from utils.triangle import Triangle
import numpy as np


class TransformMatrix(object):
    """docstring for Transform Matrix"""

    def __init__(self):
        self.dim = 4
        self._matrix = self.set_identity()

    def set_identity(self):
        return np.identity(self.dim)

    def copy_matrix(self, src):
        return np.array(src)

    def mul_matrix(self, a, b):
        return np.dot(a, b)

    def mul_matrix_from_left(self, m):
        self._matrix = np.dot(m, self._matrix)

    def mul_matrix_from_right(self, m):
        self._matrix = np.dot(self._matrix, m)

    def rotate(self, angle, vec):
        angle *= np.pi / 180
        vec_length = np.sqrt(vec['dx']**2 + vec['dy']**2 + vec['dz']**2)
        coord = {
            'dx': [(1, 1), (1, 2), (2, 1), (2, 2)],
            'dy': [(0, 0), (0, 2), (2, 0), (2, 2)],
            'dz': [(0, 0), (0, 1), (1, 0), (1, 1)]
        }

        for key in sorted(vec.keys()):
            vec[key] *= angle / vec_length
            sin = np.sin(vec[key])
            cos = np.cos(vec[key])

            m = self.set_identity()
            m[coord[key][0]] = cos
            m[coord[key][1]] = sin
            m[coord[key][2]] = -sin
            m[coord[key][3]] = cos
            self.mul_matrix_from_left(m)

    def translate(self, vec):
        m = self.set_identity()
        m[3, 0] = vec['dx']
        m[3, 1] = vec['dy']
        m[3, 2] = vec['dz']
        self.mul_matrix_from_left(m)

    def scale(self, vec):
        m = self.set_identity()
        m[0, 0] = vec['dx']
        m[1, 1] = vec['dy']
        m[2, 2] = vec['dz']
        self.mul_matrix_from_left(m)

    def shear(self, vec):
        pass

    def transform(self, vertex):
        v = np.array([vertex['x'], vertex['y'], vertex['z'], 1])
        t_v = np.dot(self._matrix.T, v)
        factor = 1/t_v[-1]
        return Triangle.Vertex(t_v[0]/factor, t_v[1]/factor, t_v[2]/factor)
