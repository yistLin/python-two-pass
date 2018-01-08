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
            b = Barrel()
            b.deserialize()
            return b
        elif entity_name == 'cuboid':
            c = Cuboid()
            c.deserialize()
            return c
        elif entity_name == 'globe':
            g = Globe()
            g.deserialize()
            return g
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

    def __init__(self, spec=0.6, refl=0.2, refr=0.1):
        super(Barrel, self).__init__(spec, refl, refr)
        self.name = "barrel"

    def deserialize(self):
        self.trianglenize()

    def trianglenize(self):
        PLANE_COUNT = 20
        RADIUS = 0.5
        four_vertex = [Triangle.Vertex() for i in range(4)]

        for i in range(PLANE_COUNT + 1):
            four_vertex[3] = deepcopy(four_vertex[2])
            four_vertex[0] = deepcopy(four_vertex[1])
            # compute coords (http://en.wikipedia.org/wiki/Sphere)
            angle = 2 * np.pi / PLANE_COUNT * i
            x, z = RADIUS * np.cos(angle), RADIUS * np.sin(angle)
            four_vertex[1]['x'] = deepcopy(x)
            four_vertex[1]['y'] = +0.5
            four_vertex[1]['z'] = deepcopy(z)
            four_vertex[2]['x'] = deepcopy(x)
            four_vertex[2]['y'] = -0.5
            four_vertex[2]['z'] = deepcopy(z)

            if i != 0:
                # face
                self.add_quad(four_vertex)
                # upper and lower base
                self.add_base_triangle(four_vertex)

    def add_base_triangle(self, four_vertex):
        t = Triangle()
        v = Triangle.Vertex()
        self.set_triangle_properties(t)

        for j in range(0, len(four_vertex), 2):
            for i in range(j + 0, j + 2):
                t.vertex[i - j] = deepcopy(four_vertex[2 * j + 1 - i])
                v['x'] = v['z'] = 0
                v['y'] = four_vertex[i]['y']
                t.vertex[i + 1 - j] = deepcopy(v)
            self.add_triangle(t)

    def add_quad(self, four_vertex):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertex[0] = deepcopy(four_vertex[0])
        t.vertex[1] = deepcopy(four_vertex[1])
        t.vertex[2] = deepcopy(four_vertex[2])
        self.add_triangle(t)

        # add triangle 2
        t.vertex[1] = deepcopy(four_vertex[2])
        t.vertex[2] = deepcopy(four_vertex[3])
        self.add_triangle(t)


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

    def deserialize(self):
        self.trianglenize()

    def trianglenize(self):
        PLANE_COUNT = 20
        RADIUS = 0.5

        lastV = [Triangle.Vertex() for i in range(PLANE_COUNT + 3)]
        thisV, now = Triangle.Vertex(), Triangle.Vertex()

        four_vertex = [Triangle.Vertex() for i in range(4)]
        for j in range(PLANE_COUNT / 2 + 1):  # for j
            angle = 2 * np.pi / PLANE_COUNT * j
            x0, z0 = RADIUS * np.cos(angle), RADIUS * np.sin(angle)
            for i in range(PLANE_COUNT + 3):  # for i
                angle2 = 2 * np.pi / PLANE_COUNT * i
                x, z = z0 * np.cos(angle2), z0 * np.sin(angle2)

                now['x'] = deepcopy(x)
                now['y'] = deepcopy(y)
                now['z'] = deepcopy(z)

                if j == 0:
                    lastV[i]['x'] = lastV[i]['z'] = 0
                    lastV[i]['y'] = RADIUS
                else:  # j != 0
                    if i == 0:
                        thisV = deepcopy(now)
                    else:  # i != 0
                        # present
                        four_vertex[0] = deepcopy(now)
                        # last
                        four_vertex[1] = deepcopy(thisV)
                        # from last pass
                        four_vertex[2] = deepcopy(lastV[i-1])
                        four_vertex[3] = deepcopy(lastV[i])
                        # fix of floating point error
                        if j == PLANE_COUNT / 2:
                            v[0]['x'], v[0]['z'] = 0, 0
                            v[1]['x'], v[1]['z'] = 0, 0

                        if (i != PLANE_COUNT+1+1) and (i != 1):
                            self.add_quad(four_vertex)

                        lastV[i-1] = thisV
                        thisV = now

    def add_quad(self, four_vertex):
        t = Triangle()
        self.set_triangle_properties(t)

        # add triangle 1
        t.vertex[0] = deepcopy(four_vertex[0])
        t.vertex[1] = deepcopy(four_vertex[1])
        t.vertex[2] = deepcopy(four_vertex[2])
        if t.vertex[0] != t.vertex[1]:  # bottom 0
            self.add_triangle(t)

        # add triangle 2
        t.vertex[1] = deepcopy(four_vertex[2])
        t.vertex[2] = deepcopy(four_vertex[3])
        if t.vertex[1] != t.vertex[2]:  # top 0
            self.add_triangle(t)


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
