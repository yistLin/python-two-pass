#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import xml.etree.ElementTree as ET

from utils import Triangle


xml_tag_name = {
    "Root": "scene",
    "Definition": "head",
    "Instantiate": "body",
    "ObjectDefinition": "objectdef",
    "Cuboid": "cuboid",
    "Barrel": "barrel",
    "Globe": "globe",
    "Teapot": "teapot"
    "TriangleSetNode": "triangleset",
    "TriangleNode": "triangle",
    "VertexNode": "vertex",
    "Rotate": "rotate",
    "Scale": "scale",
    "Shear": "shear",
    "Translate": "translate",
    "Object": "object",
    "Trianglenext": "trianglenext"
}

xml_tag_attr_name = {
    "Name": "name",
    "Emission": "emission",
    "Reflectivity": "reflectivity",
    "Radiosity": "radiosity",
    "VertexX": "x",
    "VertexY": "y",
    "VertexZ": "z",
    "TransformationX": "dx",
    "TransformationY": "dy",
    "TransformationZ": "dz",
    "RotateAngle": "angle",
    "IncludeFile": "file",
    "Spec": "spec",
    "Refr": "refr",
    "Refl": "refl"
}


def xml_read_scene(fname):
    # parse XML
    tree = ET.parse(fname)
    root = tree.getroot()

    # get objects definitions & objects translation+rotation info
    objs_root = root.find('head')
    objs_info = root.find('body')

    for obj in objs_root:
        print('==={}==={}==='.format(obj.tag, obj.attrib['name']))

        for tri in obj:
            print(tri.tag, tri.attrib)


def xml_read_tri(fname):
    # important info
    mat_c = []
    mat_e = []
    mat_p = []
    mat_spec = []
    mat_refl = []
    mat_refr = []

    # parse XML
    tree = ET.parse(fname)
    root = tree.getroot()

    # get object definitions
    obj_root = root.find('head').find('objectdef')

    # loop through objects
    for obj in obj_root:

        # loop through triangles
        for tri in obj:
            emission = tri.attrib['emission'].split(',')
            radiosity = tri.attrib['radiosity'].split(',')
            spec = tri.attrib['spec']
            refl = tri.attrib['refl']
            refr = tri.attrib['refr']
            mat_c.append([float(r) for r in radiosity])
            mat_e.append([float(e) for e in emission])
            mat_spec.append(float(spec))
            mat_refl.append(float(refl))
            mat_refr.append(float(refr))

            # loop through vertices
            vertices = []
            for vtx in tri:
                vertices.append([float(vtx.attrib['x']),
                                 float(vtx.attrib['y']),
                                 float(vtx.attrib['z'])])
            mat_p.append(vertices)

    return np.array(mat_c, dtype=np.float32), \
           np.array(mat_p, dtype=np.float32), \
           np.array(mat_e, dtype=np.float32), \
           np.array(mat_spec, dtype=np.float32), \
           np.array(mat_refl, dtype=np.float32), \
           np.array(mat_refr, dtype=np.float32)


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

