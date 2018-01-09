#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np


class Triangle(object):
    """docstring for Triangle"""

    def __init__(self, **kwargs):
        self.vertex = [Triangle.Vertex(*kwargs.get('v0', (0., 0., 0.))),
                       Triangle.Vertex(*kwargs.get('v1', (0., 0., 0.))),
                       Triangle.Vertex(*kwargs.get('v2', (0., 0., 0.)))]
        self.emission = Triangle.Color(*kwargs.get('emission', (0., 0., 0.)))
        self.reflectivity = Triangle.Color(
            *kwargs.get('reflectivity', (0., 0., 0.)))
        self.radiosity = Triangle.Color(*kwargs.get('radiosity', (0., 0., 0.)))
        self.radiosity_last = Triangle.Color(
            *kwargs.get('radiosity_last', (0., 0., 0.)))
        self.spec = kwargs.get('spec', 0.)
        self.refl = kwargs.get('refl', 0.)
        self.refr = kwargs.get('refr', 0.)
        self._vertices = np.array([Triangle.get_vertex_np(self.vertex[i])
                                    for i in range(3)])

    def __repr__(self):
        ret = []
        ret.append("Vertex: ({}, {}, {}), ({}, {}, {}), ({}, {}, {})".format(
            self.vertex[0]['x'], self.vertex[0]['y'], self.vertex[0]['z'],
            self.vertex[1]['x'], self.vertex[1]['y'], self.vertex[1]['z'],
            self.vertex[2]['x'], self.vertex[2]['y'], self.vertex[2]['z']))
        ret.append("Emission: {}".format(tuple(self.emission.values())))
        ret.append("Reflectivity: {}".format(
            tuple(self.reflectivity.values())))
        ret.append("Radiosity: {}".format(tuple(self.radiosity.values())))
        ret.append("RadiosityLast: {}".format(
            tuple(self.radiosity_last.values())))
        ret.append("spec: {}, refl: {}, refr: {}".format(
            self.spec, self.refl, self.refr))

        return '\n'.join(ret)

    @property
    def vertices(self):
        self._vertices = np.array([Triangle.get_vertex_np(self.vertex[i])
                                    for i in range(3)])
        return self._vertices

    @staticmethod
    def Vertex(x=0., y=0., z=0.):
        return {'x': x, 'y': y, 'z': z}

    @staticmethod
    def get_vertex_np(v):
        return np.array([v['x'], v['y'], v['z']])

    @staticmethod
    def Color(r=0., g=0., b=0.):
        return {'r': r, 'g': g, 'b': b}

    @staticmethod
    def get_color_np(c):
        return np.array([c['r'], c['g'], c['b']])

    @staticmethod
    def set_color_from_np(nparray):
        return Triangle.Color(nparray[0], nparray[1], nparray[2])

    @staticmethod
    def Vector(ix=0., iy=0., iz=0.):
        return {'dx': ix, 'dy': iy, 'dz': iz}

    @staticmethod
    def get_vector_np(v1, v2):
        dx = v1['x'] - v2['x']
        dy = v1['y'] - v2['y']
        dz = v1['z'] - v2['z']
        return np.array([dx, dy, dz])

    @staticmethod
    def distance(v1, v2):
        v = Triangle.get_vector_np(v1, v2)
        return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    @staticmethod
    def center(v1, v2):
        x = (v1['x'] + v2['x']) / 2
        y = (v1['y'] + v2['y']) / 2
        z = (v1['z'] + v2['z']) / 2
        return Triangle.Vertex(float(x), float(y), float(z))

    @staticmethod
    def center_of(t):
        x = (t.vertex[0]['x'] + t.vertex[1]['x'] + t.vertex[2]['x']) / 3
        y = (t.vertex[0]['y'] + t.vertex[1]['y'] + t.vertex[2]['y']) / 3
        z = (t.vertex[0]['z'] + t.vertex[1]['z'] + t.vertex[2]['z']) / 3
        return Triangle.Vertex(float(x), float(y), float(z))

    @staticmethod
    def get_normal_vector_np(t):
        v1 = t.vertex[0]
        v2 = t.vertex[1]
        v3 = t.vertex[2]
        u = Triangle.get_vector_np(v2, v1)
        v = Triangle.get_vector_np(v3, v1)
        vec = np.cross(u, v)
        return np.multiply(vec, 1 / np.linalg.norm(vec))
