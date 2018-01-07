#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np

from utils.triangle import Triangle

class FormFactor(object):
    def __init__(self, edge):
        assert edge > 0
        print('Create hemicude with edge length {}'.format(edge))

        self.edge = edge
        self.edge2 = edge * 2
        self.edge1_2 = edge / 2
        self.edge3_2 = edge * 3 / 2
        self.surface_area = self.edge**2 + 4 * self.edge * self.edge1_2

        self.hemicube = np.zeros((self.edge2, self.edge2))

        for i in range(self.edge2):
            for j in range(self.edge2):
                if not ((i < self.edge1_2 or self.edge3_2 < i) and (j < self.edge1_2 or self.edge3_2 < j)):
                    x = (i + 0.5 - self.edge) / self.edge1_2
                    y = (j + 0.5 - self.edge) / self.edge1_2
                    z = 1.0

                    if x < -1:
                        z = 2 + x
                        x = -1.0
                    if x > 1:
                        z = 2 - x
                        x = 1.0
                    if y < -1:
                        z = 2 + y
                        y = -1.0
                    if y > 1:
                        z = 2 - y
                        y = 1.0

                    self.hemicube[i][j] = z / (np.pi * (x**2 + y**2 + z**2)) / self.surface_area

        # normalization
        self.hemicube /= np.sum(self.hemicube)

    def calculate_from_factor(self, patch_list):
        patch_count = len(patch_list)
        ffs = []

        visibility_test = self.visibility_hemicube()
        for i, p_i in enumerate(patch_list):
            print('[form factor] patch {}/{} ...'.format(i, patch_count))
            for j, p_j in enumerate(patch_list):
                if i == j:
                    continue

                ci = Triangle.center_of(p_i)
                cj = Triangle.center_of(p_j)

                v_ij = Triangle.get_vector_np(cj, ci)
                n = Triangle.get_normal_vector_np(p_i)
                if np.dot(v_ij, n) <= 0:
                    continue

                transform = self.get_transform_matrix(p_i)

                v0 = np.dot(transform, Triangle.get_vector_np(p_j.vertex[0], ci))
                v1 = np.dot(transform, Triangle.get_vector_np(p_j.vertex[1], ci))
                v2 = np.dot(transform, Triangle.get_vector_np(p_j.vertex[2], ci))

                print(v0, v1, v2)

                v0 = np.multiply(v0, 1 / v0[2])
                v1 = np.multiply(v1, 1 / v1[2])
                v2 = np.multiply(v2, 1 / v2[2])

                print(v0, v1, v2)
                # dis = Triangle.distance(ci, cj)


            ffs.append(np.full(patch_count, 0.5))

        return ffs

    def visibility_hemicube(self):
        return np.full((self.edge2, self.edge2), np.inf, dtype=np.dtype([('p', np.int32), ('d', np.float64)]))

    def get_transform_matrix(self, p):
        c = Triangle.center_of(p)
        x = Triangle.get_vector_np(p.vertex[0], c)
        x = np.multiply(x, 1 / np.linalg.norm(x))
        z = Triangle.get_normal_vector_np(p)
        y = np.cross(z, x)

        cos_beta = z[2]
        sin_beta = np.sqrt(1 - cos_beta**2)

        cos_alpha = -z[1] / sin_beta
        sin_alpha = z[0] / sin_beta

        cos_gamma = y[2] / cos_beta
        sin_gamma = x[2] / cos_beta

        ty = np.array([[cos_gamma, sin_gamma, 0],
                        [-sin_gamma, cos_gamma, 0],
                        [0, 0, 1]])
        tx = np.array([[1, 0, 0],
                        [0, cos_beta, sin_beta],
                        [0, -sin_beta, cos_beta]])
        tz = np.array([[cos_alpha, sin_alpha, 0],
                        [-sin_alpha, cos_alpha, 0],
                        [0, 0, 1]])

        transform = np.dot(ty, tx)
        return np.dot(transform, tz)
