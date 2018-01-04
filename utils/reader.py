#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import xml.etree.ElementTree as ET

from utils import Triangle


def xml_read_tri(fname):
    # important info
    mat_c = []
    mat_r = []
    mat_p = []

    # parse XML
    tree = ET.parse(fname)
    root = tree.getroot()

    # get object definitions
    obj_root = root.find('head').find('objectdef')

    # loop through objects
    for obj in obj_root:

        # loop through triangles
        for tri in obj:
            radiosity = tri.attrib['radiosity'].split(',')
            reflectivity = tri.attrib['reflectivity'].split(',')
            mat_c.append([float(r) for r in radiosity])
            mat_r.append([float(r) for r in reflectivity])

            # loop through vertices
            vertices = []
            for vtx in tri:
                vertices.append([float(vtx.attrib['x']),
                                 float(vtx.attrib['y']),
                                 float(vtx.attrib['z'])])
            mat_p.append(vertices)

    return np.array(mat_c, dtype=np.float32), \
           np.array(mat_r, dtype=np.float32), \
           np.array(mat_p, dtype=np.float32)


def read_tri(fname):
    tris = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line == 'Triangle':
                colors = f.readline().strip().split()
                v1 = f.readline().strip().split()
                v2 = f.readline().strip().split()
                v3 = f.readline().strip().split()
                tri = Triangle(v1, v2, v3, colors[:3], colors[3:])
                tris.append(tri)

    return tris


if __name__ == '__main__':
    import sys
    tris = read_tri(sys.argv[1])

    print(len(tris))

    for tri in tris:
        print(tri.fcolor, tri.n)

