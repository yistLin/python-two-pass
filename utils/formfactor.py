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
            for j, p_j in enumerate(patch_list):
                if i == j:
                    continue

                ci = Triangle.center_of(p_i)
                cj = Triangle.center_of(p_j)
                dis = Triangle.distance(ci, cj)

            ffs.append(np.zeros(patch_count))

        return ffs

    def visibility_hemicube(self):
        return np.full((self.edge2, self.edge2), np.inf, dtype=np.dtype([('p', np.int32), ('d', np.float64)]))
