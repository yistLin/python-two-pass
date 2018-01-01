#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np

class Triangle(object):
    def __init__(self, a, b, c, fcolor, bcolor):
        assert len(a) == 3 and len(b) == 3 and len(c) == 3 and \
               len(fcolor) == 3 and len(bcolor) == 3

        # 3 points
        self.p = np.array([a, b, c], dtype=np.float32)

        # front and back colors
        self.fcolor = np.array(fcolor, dtype=int)
        self.bcolor = np.array(bcolor, dtype=int)

        # normal vector
        self.n = np.cross(self.p[0]-self.p[1], self.p[1]-self.p[2])
        self.n = self.n / np.linalg.norm(self.n)

    def intersect(self, ray_ori, ray_drt):
        denom = np.dot(ray_drt, self.n)

        # ignore if the angle between ray and plane is too small
        if np.abs(denom) < 1e-6:
            return np.inf, None

        # distance to the plane
        dist = np.dot(self.p[0] - ray_ori, self.n) / denom
        if dist < 0:
            return np.inf, None

        # intersection point on the plane
        pnt_int = ray_ori + dist * ray_drt

        # intersection point within triangle or not
        if not self._within(pnt_int):
            return np.inf, None

        return dist, pnt_int

    def _within(self, p):

        def same_side(p1, p2, a, b):
            cp1 = np.cross(b-a, p1-a)
            cp2 = np.cross(b-a, p2-a)
            return np.dot(cp1, cp2) >= 0

        return same_side(p, self.p[0], self.p[1], self.p[2]) and \
               same_side(p, self.p[1], self.p[0], self.p[2]) and \
               same_side(p, self.p[2], self.p[0], self.p[1])

