import numpy as np


def make_rubiks_cube_geometry():
    GREY = 0x888888

    def make_face(face_plane, face_axis, face_color, face_index_offset, position_offset):
        '''
            plane: in [0, 1]
            axis:  in [0, 1, 2]
            color: 0xFFFFFF
        '''
        f_vertices = np.array([[0, 0, face_plane], [0, 1, face_plane], [1, 1, face_plane], [1, 0, face_plane]]) + position_offset
        f_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + face_index_offset
        f_colors = np.array([face_color] * 4, dtype=np.int64)
        if face_plane == 0: f_indices = np.fliplr(f_indices)
        if face_axis != 0: f_vertices = np.roll(f_vertices, face_axis)
        return f_vertices, f_indices, f_colors

    def make_cube(cube_colors, cube_index_offset, position_offset):
        cube_vertices = np.zeros((24, 3))
        cube_indices = np.zeros((12, 3), dtype=np.int64)
        cube_colors = np.zeros((24,), dtype=np.int64)
        for f in range(6):
            f2 = f * 2
            f4 = f * 4
            face_color = cube_colors[f]
            face_plane = f // 3
            face_axis = f % 3
            face_index_offset = cube_index_offset + f4
            cube_vertices[f4:f4+4], cube_indices[f2:f2+2], cube_colors[f4:f4+4] = \
                    make_face(face_plane, face_axis, face_color, face_index_offset, position_offset)
        return cube_vertices, cube_indices, cube_colors

    vertices = np.zeros((24*27,3))
    indices = np.zeros((12*27,3), dtype=np.int64)
    colors = np.zeros((24*27,), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ijk = i*9 + j*3 + k
                ijk12 = ijk*12
                ijk24 = ijk*24
                face_colors = [GREY] * 6
                position_offset = [i, j, k]
                vertices[ijk24:ijk24+24], indices[ijk12:ijk12+12], colors[ijk24:ijk24+24] = \
                        make_cube(face_colors, ijk24, position_offset)
    return vertices, indices, colors
