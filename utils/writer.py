#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from utils.triangle import Triangle

class XMLWriter(object):
    """
    Write to XML file
    """
    @staticmethod
    def write(fname, tris):
        with open(fname, 'w') as f:
            str_header = """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<scene>
    <head>
        <objectdef name=\"scene\">
            <triangleset name=\"TriangleEntity\">"""

            str_footer = """            </triangleset>
        </objectdef>
    </head>
    <body>
        <object name=\"scene\"/>
    </body>
</scene>"""

            print(str_header, file=f)

            str_elem_s = ('                '
                          '<triangle emission=\"{:f},{:f},{:f}\" '
                          'reflectivity=\"{:f},{:f},{:f}\" '
                          'radiosity=\"{:f},{:f},{:f}\" '
                          'spec=\"{:f}\" '
                          'refl=\"{:f}\" '
                          'refr=\"{:f}\">')
            str_elem_e = ('                '
                          '</triangle>')

            for tri in tris:
                emission = list(tri.get_emission())
                reflectivity = list(tri.get_reflectivity())
                radiosity = list(tri.get_radiosity())
                spec = tri.spec
                refl = tri.refl
                refr = tri.refr
                vertices = tri.vertices

                print(str_elem_s.format(emission[0], emission[1], emission[2],
                                        reflectivity[0], reflectivity[1], reflectivity[2],
                                        radiosity[0], radiosity[1], radiosity[2],
                                        spec, refl, refr), file=f)

                for vtx in vertices:
                    print('                    '
                          '<vertex '
                          'x=\"{:f}\" '
                          'y=\"{:f}\" '
                          'z=\"{:f}\"/>'.format(vtx[0], vtx[1], vtx[2]), file=f)

                print(str_elem_e, file=f)

            print(str_footer, file=f)

