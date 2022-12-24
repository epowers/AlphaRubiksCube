class TestUtils:
    def test_01_utils_import(self):
        import alpha_rubiks_cube.utils

    def test_01_utils_make_rubiks_cube_geometry(self):
        from alpha_rubiks_cube.utils import make_rubiks_cube_geometry
        make_rubiks_cube_geometry()
