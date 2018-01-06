#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import math
from triangle import Triangle

class TriangleSet(object):
    """docstring for TriangleSet"""
    def __init__(self):
        self.triangle_set = []

    def add_triangle(self, t):
        self.triangle_set.append(t)

    def add_triangle_set(self, tset):
        for t in tset.triangle_set:
            self.triangle_set.append(t)

    def count():
        return len(self.triangle_set)

    @staticmethod
    def distance(v1, v2):
        dx = v1['x'] - v2['x']
        dy = v1['y'] - v2['y']
        dz = v1['z'] - v2['z']
        return math.sqrt(dx**2 + dy**2 + dz**2)

    @staticmethod
    def center(v1, v2):
        x = (v1['x'] + v2['x']) / 2
        y = (v1['y'] + v2['y']) / 2
        z = (v1['z'] + v2['z']) / 2
        return Triangle.Vertex(float(x), float(y), float(z))
