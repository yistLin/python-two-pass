#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import queue

import numpy as np

from utils import FormFactor
from utils import TriangleSet, Triangle

def divide(p):
    to_patch_list = []

    m0 = Triangle.center(p.vertex[0], p.vertex[1])
    m1 = Triangle.center(p.vertex[1], p.vertex[2])
    m2 = Triangle.center(p.vertex[2], p.vertex[0])

    # Triangle()

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

        area = np.sqrt(s * (s - s0) * (s - s1) * (s- s2))
        # if area > threshold:
        #     for d_p in divide(p)
        #         q.put(d_p)
        # else:
        #     to_patch_list.append(p)

    return to_patch_list

def radiosity(args):
    patch_list = TriangleSet().get_patches()

    print('{} patches'.format(len(patch_list)))
    patch_list = meshing(patch_list, args.meshing_size)
    print('{} patches'.format(len(patch_list)))

    ffs = FormFactor(args.hemicude_edge).calculate_from_factor(patch_list)

    patch_count = len(patch_list)
    for step in range(args.iter_times):
        b = np.array([Triangle.get_color_np(p.radiosity) for p in patch_list])

        for i, p in enumerate(patch_list):
            rad = np.sum(np.multiply(ffs[i], b), axis=0)
            rad = np.multiply(rad, Triangle.get_color_np(p.reflectivity))
            rad = np.add(rad, Triangle.get_color_np(p.emission))

            # patch_list[i].set radiosity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classical radiosity')
    parser.add_argument('--input_file', type=str, required=True, help='input XML path')
    parser.add_argument('--output_file', type=str, default='output.xml', help='output XML path')
    parser.add_argument('--meshing_size', type=int, default=1, help='maximum size after meshing')
    parser.add_argument('--hemicude_edge', type=int, default=256, help='hemicude edge length')
    parser.add_argument('--iter_times', type=int, default=10, help='iterate times')
    args = parser.parse_args()

    radiosity(args)
