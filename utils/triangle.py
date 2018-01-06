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
        print("spec: {}, refl: {}, refr: {}".format(self.spec, self.refl, self.refr))

    @staticmethod
    def Vertex(ix=0., iy=0., iz=0.):
        return {'x': ix, 'y': iy, 'z': iz}

    @staticmethod
    def Color(r=0., g=0., b=0.):
        return {'r': r, 'g': g, 'b': b}

    @staticmethod
    def center_of(t):
        x = (t.vertex[0]['x'] + t.vertex[1]['x'] + t.vertex[2]['x']) / 3
        y = (t.vertex[0]['y'] + t.vertex[1]['y'] + t.vertex[2]['y']) / 3
        z = (t.vertex[0]['z'] + t.vertex[1]['z'] + t.vertex[2]['z']) / 3
        return Triangle.Vertex(float(x), float(y), float(z))
