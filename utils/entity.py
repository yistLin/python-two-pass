#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import json
from copy import deepcopy
from utils.triangle import Triangle
from utils.triangle_set import TriangleSet
from utils.transform_matrix import TransformMatrix


class Entity(object):
    """docstring for Entity"""

    def __init__(self, list_of_args, name="entity", obj_name="NoObj"):
        super(Entity, self).__init__()

        raytracing_attr = {
            'entity': {'spec': 1.0, 'refl': 0.0, 'refr': 0.0},
            'barrel': {'spec': 0.6, 'refl': 0.0, 'refr': 0.1},
            'cuboid': {'spec': 1.0, 'refl': 0.0, 'refr': 0.0},
            'globe': {'spec': 0.5, 'refl': 0.0, 'refr': 0.0},
            'teapot': {'spec': 0.8, 'refl': 0.0, 'refr': 0.8},
            'tset': {'spec': 0.2, 'refl': 0.05, 'refr': 0.2},
        }

        init_attr = list_of_args[0]  # get the first element to set attr
        self.emission = np.array(init_attr.get('emission', [0., 0., 0.]))
        self.reflectivity = np.array(init_attr.get('reflectivity', [0., 0., 0.]))
        self.radiosity = np.array(init_attr.get('radiosity', [0., 0., 0.]))

        self._spec = init_attr.get('spec', raytracing_attr[name]['spec'])
        self._refl = init_attr.get('refl', raytracing_attr[name]['refl'])
        self._refr = init_attr.get('refr', raytracing_attr[name]['refr'])

        self.name = name
        self.obj_name = obj_name
        self.transform_matrix = TransformMatrix()
        self.triangle_set = TriangleSet()

    @staticmethod
    def create(entity_name, list_of_args, obj_name):
        if entity_name == 'barrel':
            b = Barrel(list_of_args, obj_name)
            b.trianglenize()
            return b
        elif entity_name == 'cuboid':
            c = Cuboid(list_of_args, obj_name)
            c.trianglenize()
            return c
        elif entity_name == 'globe':
            g = Globe(list_of_args, obj_name)
            g.trianglenize()
            return g
        elif entity_name == 'teapot':
            t = Teapot(list_of_args, obj_name)
            t.trianglenize()
            return t
        elif entity_name == 'triangleset':
            tset = Tset(list_of_args, obj_name)
            return tset
        else:
            raise NameError("There's no such entity name {}!!!".format(entity_name))

    def transform(self, trans):
        if isinstance(trans, tuple):
            trans = [trans]

        for op, param in trans:
            if op == 'rotate':
                self.transform_matrix.rotate(**param)
            elif op == 'translate':
                self.transform_matrix.translate(**param)
            elif op == 'scale':
                self.transform_matrix.scale(**param)

        self.triangle_set = self.transform_matrix.transform(self.triangle_set)

    def trianglenize(self):
        raise NotImplementedError("trianglenize() should be implemented in class inheriting Enity")

    def set_triangle_properties(self, t):
        t.emission = self.emission
        t.reflectivity = self.reflectivity
        t.radiosity = self.radiosity

        t.spec = self._spec
        t.refl = self._refl
        t.refr = self._refr

    def add_triangle(self, t):
        t = self.transform_matrix.transform(t)
        self.triangle_set.add_triangle(t)


class Barrel(Entity):
    """docstring for Barrel"""

    def __init__(self, list_of_args, obj_name):
        super(Barrel, self).__init__(list_of_args, "barrel", obj_name)

    def trianglenize(self):
        PLANE_COUNT = 20
        RADIUS = 0.5
        four_vertices = np.zeros((4, 3))

        for i in range(PLANE_COUNT + 1):
            four_vertices[3] = np.array(four_vertices[2])
            four_vertices[0] = np.array(four_vertices[1])
            # compute coords (http://en.wikipedia.org/wiki/Sphere)
            angle = 2 * np.pi / PLANE_COUNT * i
            x, z = RADIUS * np.cos(angle), RADIUS * np.sin(angle)
            four_vertices[1] = np.array([x, +0.5, z])
            four_vertices[2] = np.array([x, -0.5, z])

            if i != 0:
                # face
                self.add_quad(four_vertices)
                # upper and lower base
                self.add_base_triangle(four_vertices)

    def add_quad(self, four_vertices):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertices = np.array(four_vertices[:-1])
        self.add_triangle(deepcopy(t))

        # add triangle 2
        t.vertices[-2:] = np.array(four_vertices[2:])
        self.add_triangle(deepcopy(t))

    def add_base_triangle(self, four_vertices):
        t = Triangle()
        self.set_triangle_properties(t)

        for j in range(0, len(four_vertices), 2):
            for i in range(j + 0, j + 2):
                t.vertices[i - j] = np.array(four_vertices[2 * j + 1 - i])
                t.vertices[i + 1 - j] = np.array([0, four_vertices[i][1], 0])
            self.add_triangle(deepcopy(t))


class Cuboid(Entity):
    """docstring for Cuboid"""

    def __init__(self, list_of_args, obj_name):
        super(Cuboid, self).__init__(list_of_args, "cuboid", obj_name)

    def trianglenize(self):
        planes = [
            # left plane
            np.array([[-0.5, -0.5, -0.5],
                      [-0.5, -0.5, +0.5],
                      [-0.5, +0.5, +0.5],
                      [-0.5, +0.5, -0.5]]),
            # right plane
            np.array([[+0.5, -0.5, -0.5],
                      [+0.5, +0.5, -0.5],
                      [+0.5, +0.5, +0.5],
                      [+0.5, -0.5, +0.5]]),
            # down plane
            np.array([[-0.5, -0.5, -0.5],
                      [+0.5, -0.5, -0.5],
                      [+0.5, -0.5, +0.5],
                      [-0.5, -0.5, +0.5]]),
            # up plane
            np.array([[-0.5, +0.5, -0.5],
                      [-0.5, +0.5, +0.5],
                      [+0.5, +0.5, +0.5],
                      [+0.5, +0.5, -0.5]]),
            # rear plane
            np.array([[-0.5, -0.5, -0.5],
                      [-0.5, +0.5, -0.5],
                      [+0.5, +0.5, -0.5],
                      [+0.5, -0.5, -0.5]]),
            # front plane
            np.array([[-0.5, -0.5, +0.5],
                      [+0.5, -0.5, +0.5],
                      [+0.5, +0.5, +0.5],
                      [-0.5, +0.5, +0.5]]),
        ]

        for plane in planes:
            self.add_quad(plane)

    def add_quad(self, plane):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertices = np.array(plane[:-1])
        self.add_triangle(deepcopy(t))

        # add triangle 2
        t.vertices[-2:] = np.array(plane[2:])
        self.add_triangle(deepcopy(t))


class Globe(Entity):
    """docstring for Globe"""

    def __init__(self, list_of_args, obj_name):
        super(Globe, self).__init__(list_of_args, "globe", obj_name)

    def trianglenize(self):
        PLANE_COUNT = 20
        RADIUS = 0.5

        lastV = np.zeros((PLANE_COUNT+3, 3))

        for j in range(PLANE_COUNT // 2 + 1):  # for j
            angle = 2 * np.pi / PLANE_COUNT * j
            x0, z0 = RADIUS * np.cos(angle), RADIUS * np.sin(angle)
            for i in range(PLANE_COUNT + 3):  # for i
                angle2 = 2 * np.pi / PLANE_COUNT * i
                x, z = z0 * np.cos(angle2), z0 * np.sin(angle2)

                thisV = np.zeros(3)
                now = np.array(x, x0, z)

                if j == 0:
                    lastV[i] = np.array(0, RADIUS, 0)
                else:  # j != 0
                    if i == 0:
                        thisV = np.array(now)
                    else:  # i != 0
                        # 0: present, 1: last, 2&3: from last pass
                        four_vertices = np.array([now, thisV, lastV[i-1], lastV[i]])
                        # fix of floating point error
                        if j == PLANE_COUNT // 2:
                            four_vertices[0][0], four_vertices[0][2] = 0, 0
                            four_vertices[1][0], four_vertices[1][2] = 0, 0

                        if (i != PLANE_COUNT+1+1) and (i != 1):
                            self.add_quad(four_vertices)

                        lastV[i-1] = np.array(thisV)
                        thisV = np.array(now)

    def add_quad(self, four_vertices):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertices = np.array(four_vertices[:-1])
        if t.vertices[0] != t.vertices[1]:  # bottom 0
            self.add_triangle(deepcopy(t))

        # add triangle 2
        t.vertices[-2:] = np.array(plane[2:])
        if t.vertices[1] != t.vertices[2]:  # top 0
            self.add_triangle(deepcopy(t))


class Teapot(Entity):
    """docstring for Teapot"""

    def __init__(self, list_of_args, obj_name):
        super(Teapot, self).__init__(list_of_args, "teapot", obj_name)
        self.teapot_v = self.read('vertices')
        self.teapot_i = self.read('index')

    def read(self, indicator, fname="teapot_def.json"):
        with open(fname, 'r') as f:
            try:
                data = json.load(f)
            except:
                raise IOError("Load teapeat data error.")

        if indicator == 'vervices':
            return data['teapot_v']
        elif indicator == 'index':
            return data['teapot_i']
        else:
            raise NameError("There's no such indicator {}".format(indicator))

    def trianglenize(self):
        t = Triangle()
        self.set_triangle_properties(t)
        for i, item in enumerate(self.teapot_i):
            for j in range(3):
                t.vertices[j] = np.array(self.teapot_v[self.teapot_i[i][2-j]])
                t.vertices[j][1], t.vertices[j][2] = t.vertices[j][2], t.vertices[j][1]

            self.add_triangle(deepcopy(t))


class Tset(Entity):
    """docstring for Tset"""

    def __init__(self, list_of_args, obj_name):
        super(Tset, self).__init__(list_of_args, "tset", obj_name)
        attrs = {
            'spec': self._spec,
            'refl': self._refl,
            'refr': self._refr
        }
        for tri in list_of_args:
            self.add_triangle(Triangle(**{**tri, **attrs}))
