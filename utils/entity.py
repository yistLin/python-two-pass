#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np


class Entity(object):
    """docstring for Entity"""
    def __init__(self, spec=1.0, refl=0.0, refr=0.0):
        super(Entity, self).__init__()
        self._spec = spec
        self._refl = refl
        self._refr = refr
        self.emission = 0.0
        self.reflectivity = 0.0
        self.radiosity = 0.0
        self.patch_set = []
        self.name = "entity"

    def create(xml_node):
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
        super(Teapot, self).__init__()
        self.name = "teapot"

