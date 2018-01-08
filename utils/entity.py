#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import sys
import numpy as np
from copy import deepcopy
from utils.triangle import Triangle
from utils.triangle_set import TriangleSet
from utils.transform_matrix import TransformMatrix
from utils.teapot_def import teapot_data


class Entity(object):
    """docstring for Entity"""

    def __init__(self, spec=1.0, refl=0.0, refr=0.0):
        super(Entity, self).__init__()

        self.emission = Triangle.Color()
        self.reflectivity = Triangle.Color()
        self.radiosity = Triangle.Color()

        self._spec = spec
        self._refl = refl
        self._refr = refr

        self.transform_matrix = TransformMatrix()
        self.triangle_set = TriangleSet()
        self.name = "entity"

    @staticmethod
    def create(entity_name, list_of_args):
        if entity_name == 'barrel':
            pass
        elif entity_name == 'cuboid':
            c = Cuboid()
            c.deserialize()
            return c
        elif entity_name == 'globe':
            pass
        elif entity_name == 'teapot':
            t = Teapot()
            t.deserialize()
            return t
        elif entity_name == 'triangleset':
            pass
        else:
            print("There's no such object name {}!!!".format(obj_name))
            sys.exit(-1)

    def set_triangle_properties(self, t):
        t.emission = self.emission
        t.reflectivity = self.reflectivity
        t.radiosity = self.radiosity

        t.spec = self._spec
        t.refl = self._refl
        t.refr = self._refr

    def add_triangle(self, t):
        for i in range(3):
            t.vertex[i] = self.transform_matrix.transform(t.vertex[i])
        self.triangle_set.add_triangle(t)


class Barrel(Entity):
    """docstring for Barrel"""

    def __init__(self, spec=1.0, refl=0.0, refr=0.0):
        super(Barrel, self).__init__(spec, refl, refr)
        self.name = "barrel"


class Cuboid(Entity):
    """docstring for Cuboid"""

    def __init__(self, spec=1.0, refl=0.1, refr=0.0):
        super(Cuboid, self).__init__(spec, refl, refr)
        self.name = "cuboid"

    def deserialize(self):
        self.trianglenize()

    def trianglenize(self):
        planes = [
            [  # left plane
                Triangle.Vertex(-0.5, -0.5, -0.5),
                Triangle.Vertex(-0.5, -0.5, +0.5),
                Triangle.Vertex(-0.5, +0.5, +0.5),
                Triangle.Vertex(-0.5, +0.5, -0.5),
            ],
            [  # right plane
                Triangle.Vertex(+0.5, -0.5, -0.5),
                Triangle.Vertex(+0.5, +0.5, -0.5),
                Triangle.Vertex(+0.5, +0.5, +0.5),
                Triangle.Vertex(+0.5, -0.5, +0.5),
            ],
            [  # down plane
                Triangle.Vertex(-0.5, -0.5, -0.5),
                Triangle.Vertex(+0.5, -0.5, -0.5),
                Triangle.Vertex(+0.5, -0.5, +0.5),
                Triangle.Vertex(-0.5, -0.5, +0.5),
            ],
            [  # up plane
                Triangle.Vertex(-0.5, +0.5, -0.5),
                Triangle.Vertex(-0.5, +0.5, +0.5),
                Triangle.Vertex(+0.5, +0.5, +0.5),
                Triangle.Vertex(+0.5, +0.5, -0.5),
            ],
            [  # rear plane
                Triangle.Vertex(-0.5, -0.5, -0.5),
                Triangle.Vertex(-0.5, +0.5, -0.5),
                Triangle.Vertex(+0.5, +0.5, -0.5),
                Triangle.Vertex(+0.5, -0.5, -0.5),
            ],
            [  # front plane
                Triangle.Vertex(-0.5, -0.5, +0.5),
                Triangle.Vertex(+0.5, -0.5, +0.5),
                Triangle.Vertex(+0.5, +0.5, +0.5),
                Triangle.Vertex(-0.5, +0.5, +0.5),
            ],
        ]

        for plane in planes:
            self.add_quad(plane)

    def add_quad(self, plane):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertex[0] = deepcopy(plane[0])
        t.vertex[1] = deepcopy(plane[1])
        t.vertex[2] = deepcopy(plane[2])
        self.add_triangle(t)

        # add triangle 2
        t.vertex[1] = deepcopy(plane[2])
        t.vertex[2] = deepcopy(plane[3])
        self.add_triangle(t)


class Globe(Entity):
    """docstring for Globe"""

    def __init__(self, spec=1.0, refl=0.7, refr=0.0):
        super(Globe, self).__init__(spec, refl, refr)
        self.name = "globe"


class Teapot(Entity):
    """docstring for Teapot"""

    def __init__(self, spec=1.0, refl=0.0, refr=0.8):
        super(Teapot, self).__init__(spec, refl, refr)
        self.name = "teapot"
        self.teapot_v = self.read('v')
        self.teapot_t = self.read('t')

    def read(self, indicator):
        if indicator == 'v':
            return teapot_data['teapot_v']
        elif indicator == 't':
            return teapot_data['teapot_t']
        else:
            print("There's no such indicator {}".format(indicator))
            sys.exit(-2)

    def deserialize(self):
        self.trianglenize()

    def trianglenize(self):
        for i, item in enumerate(self.teapot_t):
            t = Triangle()
            self.set_triangle_properties(t)

            for j in range(3):
                t.vertex[j] = deepcopy(self.teapot_v[self.teapot_t[i][2 - j]])
                t.vertex[j]['y'], t.vertex[j]['z'] = \
                    t.vertex[j]['z'], t.vertex[j]['y']

            self.add_triangle(t)
