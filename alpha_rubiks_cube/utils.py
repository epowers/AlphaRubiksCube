import numpy as np


def make_rubiks_cube_geometry():
    BLUE = 0x0000ff
    WHITE = 0xffffff
    RED = 0xff0000
    YELLOW = 0xffff00
    GREEN = 0x008000
    ORANGE = 0xffa500
    GREY = 0x888888
    CUBE_COLORS = [BLUE, WHITE, RED, GREEN, YELLOW, ORANGE]

    def make_face(face_plane, face_axis, face_color, face_index_offset):
        '''
            plane: in [0, 1]
            axis:  in [0, 1, 2]
            color: 0xFFFFFF
        '''
        f_vertices = np.array([[face_plane, 0, 0], [face_plane, 0, 1], [face_plane, 1, 1], [face_plane, 1, 0]], dtype=np.float32)
        f_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        f_colors = np.array([face_color] * 4, dtype=np.uint32)
        if face_plane != 0: f_indices = np.fliplr(f_indices)
        if face_axis != 0: f_vertices = np.roll(f_vertices, face_axis, axis=1)
        f_indices += face_index_offset
        return f_vertices, f_indices, f_colors

    def make_cube(face_colors):
        cube_vertices = np.zeros((24, 3), dtype=np.float32)
        cube_indices = np.zeros((12, 3), dtype=np.uint32)
        cube_colors = np.zeros((24,), dtype=np.uint32)
        for f in range(6):
            f2 = f * 2
            f4 = f * 4
            face_color = face_colors[f]
            face_plane = f // 3
            face_axis = f % 3
            face_index_offset = f4
            cube_vertices[f4:f4+4], cube_indices[f2:f2+2], cube_colors[f4:f4+4] = \
                    make_face(face_plane, face_axis, face_color, face_index_offset)
        return cube_vertices, cube_indices, cube_colors

    vertices = [None] * 27
    indices = [None] * 27
    colors = [None] * 27
    translations = [None] * 27
    for x in range(3):
        for y in range(3):
            for z in range(3):
                xyz = x + y*3 + z*9
                xyz12 = xyz*12
                xyz24 = xyz*24
                position_offset = [x, y, z]

                face_colors = np.array(CUBE_COLORS, dtype=np.uint32)
                for c, d in enumerate([x, y, z]):
                    if d == 0:
                        face_colors[c+3] = GREY
                    elif d == 1:
                        face_colors[c+0] = GREY
                        face_colors[c+3] = GREY
                    elif d == 2:
                        face_colors[c+0] = GREY

                vertices[xyz], indices[xyz], colors[xyz] = make_cube(face_colors)
                translations[xyz] = position_offset
    return zip(translations, vertices, indices, colors)

def k3d_generate_rubiks_cube_drawable_list(cube, state):
    import k3d
    from k3d.transform import process_transform_arguments, transform as Transform

    drawables = [None] * 27

    for i, (xyz, vertices, indices, colors) in enumerate(cube):
        # TODO: calculate rotations
        axis = [1, 0, 0]
        #angle = np.pi / 2
        angle = 0.0

        # transform
        mesh = drawables[i] = k3d.mesh(vertices, indices, colors=colors)
        translate = Transform(translation=[0.5, 0.5, 0.5])
        rotate = Transform(rotation=[angle] + axis, parent=translate)
        translate = Transform(translation=[-0.5, -0.5, -0.5], parent=rotate)
        translate = Transform(translation=xyz, parent=translate)
        transform = translate
        process_transform_arguments(mesh, transform=transform)

    return drawables
