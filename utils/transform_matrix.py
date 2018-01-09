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

            m = self.set_identity()
            m[coord[key][0]] = cos
            m[coord[key][1]] = sin
            m[coord[key][2]] = -sin
            m[coord[key][3]] = cos
            self.mul_matrix_from_left(m)

    def translate(self, dx=0., dy=0., dz=0.):
        m = self.set_identity()
        m[3, 0] = dx
        m[3, 1] = dy
        m[3, 2] = dz
        self.mul_matrix_from_left(m)

    def scale(self, dx=1., dy=1., dz=1.):
        m = self.set_identity()
        m[0, 0] = dx
        m[1, 1] = dy
        m[2, 2] = dz
        self.mul_matrix_from_left(m)

    def shear(self, vec):
        pass

    def transform(self, input):
        if np.array_equal(self._matrix, np.identity(self.dim)):
            return input

        if isinstance(input, dict):
            v = Triangle.get_vertex_np(input)
        elif isinstance(input, Triangle):
            v = input.vertices
        elif isinstance(input, TriangleSet):
            v = np.array([tri.vertices for tri in input])
        else:
            raise TypeError("input should be either Vertex,"
                            "Triangle or TriangleSet.")

        if len(v.shape) != 2:
            v = v.reshape(-1, 3)

        v = self._transform_utils(v)

        if isinstance(input, dict):
            v = v.squeeze()
            output = Triangle.Vertex(v[0], v[1], v[2])
        elif isinstance(input, Triangle):
            output = deepcopy(input)
            output.vertex = [Triangle.Vertex(*v[i]) for i in range(3)]
        elif isinstance(input, TriangleSet):
            v = v.reshape(len(input), 3, 3)
            output = deepcopy(input)
            for i in range(len(input)):
                output.triangle_set[i].vertex = [Triangle.Vertex(*v[i, j, :])
                                                    for j in range(3)]

        return output

    def _transform_utils(self, v):
        v_homo = np.ones((v.shape[0], v.shape[1] + 1))
        v_homo[:, :-1] = v
        t_v = np.dot(self._matrix.T, v_homo.T)
        factor = (1 / t_v.T[:, -1]).reshape(-1, 1)

        return t_v.T[:, :-1] / factor
