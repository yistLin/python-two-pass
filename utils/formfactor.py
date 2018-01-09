#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from multiprocessing import Pool

import numpy as np
from numpy.core.umath_tests import inner1d

from utils.triangle import Triangle

class FormFactor(object):
    def __init__(self, args, patch_list):
        assert args.hemicube_edge > 0
        print('Create hemicude with edge length {}'.format(args.hemicube_edge))

        self.edge = args.hemicube_edge
        self.edge2 = args.hemicube_edge * 2
        self.edge1_2 = args.hemicube_edge / 2
        self.edge3_2 = args.hemicube_edge * 3 / 2
        self.surface_area = self.edge**2 + 4 * self.edge * self.edge1_2

        self.patch_list = patch_list
        self.patch_count = len(patch_list)

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

    def calculate_form_factor(self, processes):
        with Pool(processes=processes) as pool:
            ffs = pool.starmap(self.calculate_one_patch_form_factor, enumerate(self.patch_list))

        return np.array(ffs)

    def calculate_one_patch_form_factor(self, i, p_i):
        print('[form factor] patch {}/{} ...'.format(i, self.patch_count))

        N_vs = []
        N_distance = []
        for j, p_j in enumerate(self.patch_list):
            if i == j:
                N_vs.append(np.array([[-1., -1.], [-1., -2.], [-2., -1.]]))
                N_distance.append(np.inf)
                continue

            ci = Triangle.center_of(p_i)
            cj = Triangle.center_of(p_j)

            v_ij = Triangle.get_vector_np(cj, ci)
            n = Triangle.get_normal_vector_np(p_i)
            if np.dot(v_ij, n) <= 0:
                N_vs.append(np.array([[-1., -1.], [-1., -2.], [-2., -1.]]))
                N_distance.append(np.inf)
                continue

            transform = self.get_transform_matrix(p_i)
            v0 = np.dot(Triangle.get_vector_np(p_j.vertex[0], ci), transform)
            v1 = np.dot(Triangle.get_vector_np(p_j.vertex[1], ci), transform)
            v2 = np.dot(Triangle.get_vector_np(p_j.vertex[2], ci), transform)
            v0 = self.project(v0)
            v1 = self.project(v1)
            v2 = self.project(v2)
            vs = np.array([v0, v1, v2])
            N_vs.append(vs)

            N_distance.append(Triangle.distance(ci, cj))

        # collect all triangles and distances
        assert len(N_vs) == self.patch_count
        assert len(N_distance) == self.patch_count
        N_vs = np.array(N_vs)
        N_distance = np.array(N_distance)

        # create empty form factor array
        ff = np.zeros(self.patch_count)

        # speed up intersection test
        N_v0 = N_vs[:, 2] - N_vs[:, 0]
        N_v1 = N_vs[:, 1] - N_vs[:, 0]
        d00 = inner1d(N_v0, N_v0)
        d01 = inner1d(N_v0, N_v1)
        d11 = inner1d(N_v1, N_v1)
        invDenom = 1. / (d00 * d11 - d01 * d01)

        # iterate hemicube
        for x in range(self.edge2):
            for y in range(self.edge2):
                # Barycentric Technique
                pnt_int = np.array([x + 0.5, y + 0.5])
                N_v2 = pnt_int - N_vs[:, 0]
                d02 = inner1d(N_v0, N_v2)
                d12 = inner1d(N_v1, N_v2)
                u = (d11 * d02 - d01 * d12) * invDenom
                v = (d00 * d12 - d01 * d02) * invDenom

                # inside triangle
                within = (u >= 0.) & (v >= 0.) & (u + v < 1.)
                if within.any():
                    distance = N_distance.copy()
                    distance[~within] = np.inf
                    ff[np.argmin(distance)] += self.delta_formfactor[x][y]

        return ff

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
        if z < 0:
            z = 0.0

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
