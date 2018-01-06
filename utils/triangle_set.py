#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from utils.triangle import Triangle


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
