#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import queue

import numpy as np

from utils import FormFactor

def divide():
    to_patch_list = []

    return to_patch_list

def meshing(from_patch_list, threshold):
    print("meshing... with threshold {}".format(threshold))

    q = queue.Queue()
    for p in from_patch_list:
        q.put(p)

    to_patch_list = []
    while not q.empty():
        p = q.get()

        # area =
        # if area > threshold:
        #     for d_p in divide(p)
        #         q.put(d_p)
        # else:
        #     to_patch_list.append(p)

    return to_patch_list

def radiosity(args):
    patch_list = []

    print('{} patches'.format(len(patch_list)))
    patch_list = meshing(patch_list, args.meshing_size)
    print('{} patches'.format(len(patch_list)))

    FormFactor(args.hemicude_edge).calculate_from_factor(patch_list)

    patch_count = len(patch_list)
    for i in range(args.iter_times):
        b = np.zeros(patch_count)

        for i, p in enumerate(patch_list):
            b[i] = p.rad

        for p in patch_list:
            rad = np.dot(p.ff, b)
            rad *= p.refl
            rad += p.emission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classical radiosity')
    parser.add_argument('--input_file', type=str, required=True, help='input XML path')
    parser.add_argument('--output_file', type=str, default='output.xml', help='output XML path')
    parser.add_argument('--meshing_size', type=int, default=1, help='maximum size after meshing')
    parser.add_argument('--hemicude_edge', type=int, default=256, help='hemicude edge length')
    parser.add_argument('--iter_times', type=int, default=10, help='iterate times')
    args = parser.parse_args()

    radiosity(args)
