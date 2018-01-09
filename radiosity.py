#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import queue

import numpy as np

from utils import FormFactor
from utils import TriangleSet, Triangle
from utils.reader import xml_read_scene
from utils import XMLWriter

def divide(p):
    to_patch_list = []

    m0 = Triangle.center(p.vertex[0], p.vertex[1])
    m1 = Triangle.center(p.vertex[1], p.vertex[2])
    m2 = Triangle.center(p.vertex[2], p.vertex[0])

    to_patch_list.append(Triangle(v0=(p.vertex[0]['x'], p.vertex[0]['y'], p.vertex[0]['z']), v1=(m0['x'], m0['y'], m0['z']), v2=(m2['x'], m2['y'], m2['z']), emission=p.get_emission(), reflectivity=p.get_reflectivity(), spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.append(Triangle(v0=(p.vertex[1]['x'], p.vertex[1]['y'], p.vertex[1]['z']), v1=(m1['x'], m1['y'], m1['z']), v2=(m0['x'], m0['y'], m0['z']), emission=p.get_emission(), reflectivity=p.get_reflectivity(), spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.append(Triangle(v0=(p.vertex[2]['x'], p.vertex[2]['y'], p.vertex[2]['z']), v1=(m2['x'], m2['y'], m2['z']), v2=(m1['x'], m1['y'], m1['z']), emission=p.get_emission(), reflectivity=p.get_reflectivity(), spec=p.spec, refl=p.refl, refr=p.refr))
    to_patch_list.append(Triangle(v0=(m0['x'], m0['y'], m0['z']), v1=(m1['x'], m1['y'], m1['z']), v2=(m2['x'], m2['y'], m2['z']), emission=p.get_emission(), reflectivity=p.get_reflectivity(), spec=p.spec, refl=p.refl, refr=p.refr))

    return to_patch_list

def meshing(from_patch_list, threshold):
    print("meshing... with threshold {}".format(threshold))

    q = queue.Queue()
    for p in from_patch_list:
        q.put(p)

    to_patch_list = []
    while not q.empty():
        p = q.get()

        s0 = Triangle.distance(p.vertex[0], p.vertex[1])
        s1 = Triangle.distance(p.vertex[1], p.vertex[2])
        s2 = Triangle.distance(p.vertex[2], p.vertex[0])

        s = (s0 + s1 + s2) / 2

        area = np.sqrt(s * (s - s0) * (s - s1) * (s - s2))
        if area > threshold:
            for d_p in divide(p):
                q.put(d_p)
        else:
            to_patch_list.append(p)

    return to_patch_list

def radiosity(args):
    patch_list = []
    scene = xml_read_scene(args.input_file)
    for name, e in scene.items():
        if 'teapot' not in name:
            patch_list.extend(e.triangle_set.get_patches())

    print('Total {} patches'.format(len(patch_list)))
    patch_list = meshing(patch_list, args.meshing_size)
    print('Total {} patches'.format(len(patch_list)))

    ffs = FormFactor(args, patch_list).calculate_form_factor(args.processes)
    np.save('ffs-{}-m{}-h{}'.format(args.input_file.split('/')[-1], args.meshing_size, args.hemicube_edge), ffs)

    for i, p in enumerate(patch_list):
        patch_list[i].set_radiosity(patch_list[i].get_emission())

    patch_count = len(patch_list)
    for step in range(args.iter_times):
        print('step {}/{}'.format(step + 1, args.iter_times))
        b = np.array([Triangle.get_color_np(p.radiosity) for p in patch_list])

        for i, p in enumerate(patch_list):
            rad = np.sum(np.multiply(b, ffs[i][:, np.newaxis]), axis=0)
            rad = np.multiply(rad, Triangle.get_color_np(p.reflectivity))
            rad = np.add(rad, Triangle.get_color_np(p.emission))
            patch_list[i].set_radiosity(rad)

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
    args = parser.parse_args()

    radiosity(args)
