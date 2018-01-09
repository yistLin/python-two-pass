#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np


class Triangle(object):
    """docstring for Triangle

        vertices: np.ndarray, shape=(3, 3)
        emission: np.ndarray, shape=(3,)
        reflectivity: np.ndarray, shape=(3,)
        radiosity: np.ndarray, shape=(3,)
        radiosity_last: np.ndarray, shape=(3,)
        spec: float
        refl: float
        refr: float
    """

    def __init__(self, **kwargs):
        self.vertices = np.array(kwargs.get('vertices', np.zeros((3, 3))))
        self.emission = np.array(kwargs.get('emission', np.zeros(3)))
        self.reflectivity = np.array(kwargs.get('reflectivity', np.zeros(3)))
        self.radiosity = np.array(kwargs.get('radiosity', np.zeros(3)))
        self.radiosity_last = np.array(kwargs.get('radiosity_last', np.zeros(3)))
        self.spec = 0.
        self.refl = 0.
        self.refr = 0.

    def __repr__(self):
        ret = []
        ret.append("Vertices: {}, {}, {}".format(
            self.vertices[0], self.vertices[1], self.vertices[2]))
        ret.append("Emission: {}".format(self.emission))
        ret.append("Reflectivity: {}".format(self.reflectivity))
        ret.append("Radiosity: {}".format(self.radiosity))
        ret.append("RadiosityLast: {}".format(self.radiosity_last))
        ret.append("spec: {}, refl: {}, refr: {}".format(
            self.spec, self.refl, self.refr))

        return '\n'.join(ret)

    @property
    def normal(self):
        u = vectorize(self.vertices[0], self.vertices[1])
        v = vectorize(self.vertices[0], self.vertices[2])
        vec = np.cross(u, v)
        return np.multiply(vec, 1 / np.linalg.norm(vec))

    def edge_centers(self):
        return np.array([(self.vertices[0] + self.vertices[1]) / 2,
                         (self.vertices[1] + self.vertices[2]) / 2,
                         (self.vertices[2] + self.vertices[0]) / 2])

    def center(self):
        return (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3


def vectorize(a, b):
    return b - a


def distance(a, b):
    v = vectorize(a, b)
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
