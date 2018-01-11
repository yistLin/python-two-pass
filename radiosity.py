#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import queue

import numpy as np

from utils import FormFactor
from utils import TriangleSet, Triangle, distance
from utils.reader import XMLReader
from utils import XMLWriter


def divide(p):
    to_patch_list = TriangleSet()

    m = p.edge_centers()
    m0 = m[0]
    m1 = m[1]
    m2 = m[2]

    to_patch_list.add_triangle(Triangle(vertices=np.array([p.vertices[0], m0, m2]), emission=p.emission, reflectivity=p.reflectivity, spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.add_triangle(Triangle(vertices=np.array([p.vertices[1], m1, m0]), emission=p.emission, reflectivity=p.reflectivity, spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.add_triangle(Triangle(vertices=np.array([p.vertices[2], m2, m1]), emission=p.emission, reflectivity=p.reflectivity, spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.add_triangle(Triangle(vertices=np.array(m), emission=p.emission, reflectivity=p.reflectivity, spec=p.spec, refl=p.refl, refr=p.refr))

    return to_patch_list


def meshing(from_patch_list, threshold):
    print("meshing... with threshold {}".format(threshold))

    q = queue.Queue()
    for p in from_patch_list:
        q.put(p)

    to_patch_list = TriangleSet()
    while not q.empty():
        p = q.get()

        s0 = distance(p.vertices[0], p.vertices[1])
        s1 = distance(p.vertices[1], p.vertices[2])
        s2 = distance(p.vertices[2], p.vertices[0])

        s = (s0 + s1 + s2) / 2

        area = np.sqrt(s * (s - s0) * (s - s1) * (s - s2))
        if area > threshold:
            for d_p in divide(p):
                q.put(d_p)
        else:
            to_patch_list.add_triangle(p)

    return to_patch_list


def radiosity(args):
    patch_list = TriangleSet()
    scene = XMLReader.read_scene(args.input_file)
    for name, e in scene.items():
        if 'teapot' not in name:
            patch_list.add_triangle_set(e.triangle_set)

    print('Total {} patches'.format(len(patch_list)))
    patch_list = meshing(patch_list, args.meshing_size)
    print('Total {} patches'.format(len(patch_list)))

    if args.load_ffs is None:
        ffs = FormFactor(args, patch_list).calculate_form_factor(args.processes)
        np.save('ffs-{}-m{}-h{}'.format(args.input_file.split('/')[-1], args.meshing_size, args.hemicube_edge), ffs)
    else:
        ffs = np.load(args.load_ffs)

    for i, p in enumerate(patch_list):
        patch_list[i].radiosity = np.array(patch_list[i].emission)

    patch_count = len(patch_list)
    for step in range(args.iter_times):
        print('step {}/{}'.format(step + 1, args.iter_times))
        b = np.array([p.radiosity for p in patch_list])

        for i, p in enumerate(patch_list):
            rad = np.sum(np.multiply(b, ffs[i][:, np.newaxis]), axis=0)
            rad = np.multiply(rad, p.reflectivity)
            rad = np.add(rad, p.emission)
            patch_list[i].radiosity = np.array(rad)

        XMLWriter.write('{}-step{}'.format(args.output_file, step + 1), patch_list)

    XMLWriter.write(args.output_file, patch_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classical radiosity')
    parser.add_argument('--input_file', type=str, required=True, help='input XML path')
    parser.add_argument('--output_file', type=str, default='output.xml', help='output XML path')
    parser.add_argument('--meshing_size', type=int, default=1, help='maximum size after meshing')
    parser.add_argument('--hemicube_edge', type=int, default=256, help='hemicube edge length')
    parser.add_argument('--iter_times', type=int, default=10, help='iterate times')
    parser.add_argument('--processes', type=int, default=4, help='processes')
    parser.add_argument('--load_ffs', type=str, help='load ffs')
    args = parser.parse_args()

    radiosity(args)
