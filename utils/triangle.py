#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np


class Triangle(object):
    """docstring for Triangle"""

    def __init__(self):
        self.vertex = [Triangle.Vertex(0., 0., 0.),
                       Triangle.Vertex(0., 0., 0.),
                       Triangle.Vertex(0., 0., 0.)]
        self.emission = Triangle.Color(0., 0., 0.)
        self.reflectivity = Triangle.Color(0., 0., 0.)
        self.radiosity = Triangle.Color(0., 0., 0.)
        self.radiosity_last = Triangle.Color(0., 0., 0.)
        self.spec = 0.
        self.refl = 0.
        self.refr = 0.

    def print_attr(self):
        print("Vertex: ({}, {}, {}), ({}, {}, {}), ({}, {}, {})".format(
            self.vertex[0]['x'], self.vertex[0]['y'], self.vertex[0]['z'],
            self.vertex[1]['x'], self.vertex[1]['y'], self.vertex[1]['z'],
            self.vertex[2]['x'], self.vertex[2]['y'], self.vertex[2]['z']))
        print("Emission: {}".format(self.emission))
        print("Reflectivity: {}".format(self.reflectivity))
        print("Radiosity: {}".format(self.radiosity))
        print("RadiosityLast: {}".format(self.radiosity_last))
        print("spec: {}, refl: {}, refr: {}".format(
            self.spec, self.refl, self.refr))

    @staticmethod
    def Vertex(ix=0., iy=0., iz=0.):
        return {'x': ix, 'y': iy, 'z': iz}

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
    def get_normal_vector_np(v1, v2, v3):
        u = get_vector_np(v2, v1)
        v = get_vector_np(v3, v1)
        vec = np.cross(u, v)
        vec_length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        vec /= vec_length
        return Triangle.Vertex(float(x), float(y), float(z))
