#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from utils.triangle import Triangle


class TriangleSet(object):
    """docstring for TriangleSet"""

    def __init__(self):
        self.triangle_set = []

    def __len__(self):
        return len(self.triangle_set)

    def __iter__(self):
        return iter(self.triangle_set)

    def __next__(self):
        return next(self.triangle_set)

    def add_triangle(self, t):
        self.triangle_set.append(t)

    def add_triangle_set(self, tset):
        self.triangle_set += tset.triangle_set

    def get_patches(self):
        return self.triangle_set
