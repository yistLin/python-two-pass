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

        self.delta_formfactor = np.zeros((self.edge2, self.edge2))

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

                    self.delta_formfactor[i][j] = z / (np.pi * (x**2 + y**2 + z**2)) / self.surface_area

        # normalization
        self.delta_formfactor /= np.sum(self.delta_formfactor)

    def calculate_from_factor(self, patch_list):
        patch_count = len(patch_list)
        ffs = []

        for i, p_i in enumerate(patch_list):
            print('[form factor] patch {}/{} ...'.format(i, patch_count))

            visibility_test = self.visibility_hemicube()
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

                v0 = np.dot(Triangle.get_vector_np(p_j.vertex[0], ci), transform)
                v1 = np.dot(Triangle.get_vector_np(p_j.vertex[1], ci), transform)
                v2 = np.dot(Triangle.get_vector_np(p_j.vertex[2], ci), transform)

                print('original v0, v1, v2\n', v0, v1, v2)

                v0 = self.project(v0)
                v1 = self.project(v1)
                v2 = self.project(v2)

                print('transformed v0, v1, v2\n', v0, v1, v2)

                distance = Triangle.distance(ci, cj)
                for x in range(self.edge2):
                    for y in range(self.edge2):
                        if self.check_inside((x + 0.5, y + 0.5), v0, v1, v2):
                            if visibility_test[x][y][1] > distance:
                                visibility_test[x][y][0] = j
                                visibility_test[x][y][1] = distance

            ff = np.zeros(patch_count)
            for x in range(self.edge2):
                for y in range(self.edge2):
                    j = visibility_test[x][y][0]
                    if 0 <= j < patch_count:
                        ff[j] += self.delta_formfactor[x][y]
            print('ff', ff)
            ffs.append(ff)

        return ffs

    def visibility_hemicube(self):
        return np.full((self.edge2, self.edge2), np.inf, dtype=np.dtype([('p', np.int32), ('d', np.float64)]))

    def get_transform_matrix(self, p):
        c = Triangle.center_of(p)
        x = Triangle.get_vector_np(p.vertex[0], c)
        x = np.multiply(x, 1 / np.linalg.norm(x))
        z = Triangle.get_normal_vector_np(p)
        y = np.cross(z, x)

        A = np.array([x, y, z])
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        X = np.linalg.solve(A, B)
        return X

    def project(self, v):
        x = v[0]
        y = v[1]
        z = v[2]

        # side: right = 0, up = 1, left = 2, down = 3
        side = -1
        if x >= 0 and y >= 0:
            if x > y:
                side = 0
            else:
                side = 1
        elif x < 0 and y >= 0:
            if -x > y:
                side = 2
            else:
                side = 1
        elif x < 0 and y < 0:
            if -x > -y:
                side = 2
            else:
                side = 3
        else:
            if x > -y:
                side = 0
            else:
                side = 3

        xy = np.sqrt(x**2 + y**2)
        theta = np.arctan(z / xy)
        if side == 0:
            if theta >= np.arctan(1 / np.sqrt((y / x)**2 + 1)):
                return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
            else:
                return self.edge - self.edge1_2 * (y / np.abs(x)), self.edge2 - self.edge1_2 * (z / np.abs(x))
        elif side == 1:
            if theta >= np.arctan(1 / np.sqrt((x / y)**2 + 1)):
                return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
            else:
                return self.edge1_2 * (z / np.abs(y)), self.edge + self.edge1_2 * (x / np.abs(y))
        elif side == 2:
            if theta >= np.arctan(1 / np.sqrt((y / x)**2 + 1)):
                return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
            else:
                return self.edge - self.edge1_2 * (y / np.abs(x)), self.edge1_2 * (z / np.abs(x))
        else:
            if theta >= np.arctan(1 / np.sqrt((x / y)**2 + 1)):
                return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
            else:
                return self.edge2 - self.edge1_2 * (z / np.abs(y)), self.edge + self.edge1_2 * (x / np.abs(y))

    def check_inside(self, v, v0, v1, v2):
        vec0 = np.array([v0[0] - v[0], v0[1] - v[1]])
        vec1 = np.array([v1[0] - v[0], v1[1] - v[1]])
        vec2 = np.array([v2[0] - v[0], v2[1] - v[1]])

        c0 = np.cross(vec0, vec1)
        c1 = np.cross(vec1, vec2)
        c2 = np.cross(vec2, vec0)

        if np.dot(c0, c1) < 0:
            return False
        if np.dot(c1, c2) < 0:
            return False
        return True
