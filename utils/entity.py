#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import sys
import numpy as np
from utils.triangle import Triangle
from utils.triangle_set import TriangleSet
from utils.transform_matrix import TransformMatrix


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
            pass
        elif entity_name == 'globe':
            pass
        elif entity_name == 'teapot':
            pass
        elif entity_name == 'triangleset':
            pass
        else:
            print("There's no such object name {}!!!".format(obj_name))
            sys.exit(-1)

    def set_triangle_properties(self, t):
        t.emission = self.emission
        t.reflectivity = self.reflectivity
        t.radiosity = self.radiosity

        t._spec = self._spec
        t._refl = self._refl
        t._refr = self._refr


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
        self.teapot_v = self.read()
        self.teapot_v_count = len(self.teapot_v)
        self.teapot_t = self.read()
        self.teapot_t_count = len(self.teapot_t)

    def deserialize():
        pass
