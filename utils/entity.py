#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from utils.triangle import Triangle


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

        self.transform_matrix = None
        self.triangle_set = []
        self.name = "entity"

    def create(self, xml_node):
        node_name = xml_node.name


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
