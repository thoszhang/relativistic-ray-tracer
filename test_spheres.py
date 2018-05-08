import pytest

import numpy as np
import numpy.testing as nptest
import spheres
import util


class TestSphereIntersection:
    def test_sphere_two_intersections(self):
        sphere = spheres.Sphere(1)
        intersection, color = sphere.get_intersection_and_color(spheres.Ray(np.array([0, 3, 0, 0]),
                                                                np.array([0, -1, 0, 0])))
        nptest.assert_almost_equal(intersection, np.array([1, 0, 0]))

    def test_sphere_no_intersection(self):
        sphere = spheres.Sphere(1)
        intersection, color = sphere.get_intersection_and_color(spheres.Ray(np.array([0, 2, 0, 0]),
                                                                np.array([0, 0, 1, 0])))
        assert intersection is None

    def test_ray_pointing_other_direction(self):
        sphere = spheres.Sphere(1)
        intersection, color = sphere.get_intersection_and_color(spheres.Ray(np.array([0, 0, 2, 0]),
                                                                np.array([0, 0, 1, 0])))
        assert intersection is None

    def test_ray_starts_inside_sphere(self):
        sphere = spheres.Sphere(1)
        intersection, color = sphere.get_intersection_and_color(spheres.Ray(np.array([0, 0, 0, 0]),
                                                                np.array([0, 0, 2, 0])))
        nptest.assert_almost_equal(intersection, np.array([0, 1, 0]))


class TestCylinderIntersection:
    def test_cylinder_two_intersections_orthogonal(self):
        cylinder = spheres.Cylinder((1., 0, -1.), (1., 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 3., 0, 0]),
                                                                    np.array([0, -1., 0, 0])))
        nptest.assert_almost_equal(intersection, np.array([2., 0, 0]))

    def test_cylinder_no_intersections_orthogonal(self):
        cylinder = spheres.Cylinder((1., 0, -1.), (1., 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 2., 2., 0]),
                                                                  np.array([0, -1., 0, 0])))
        assert intersection is None

    def test_cylinder_no_intersections_beyond_edge(self):
        cylinder = spheres.Cylinder((0, 0, -1.), (0, 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 2., 0, 2.]),
                                                                  np.array([0, -1., 0, 0])))
        assert intersection is None

    def test_cylinder_two_intersections_on_edge(self):
        cylinder = spheres.Cylinder((0, 0, -1.), (0, 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 2., 0, 1.]),
                                                                  np.array([0, -1., 0, 0])))
        nptest.assert_almost_equal(intersection, np.array([1., 0, 1.]))

    def test_cylinder_one_intersection_orthogonal(self):
        cylinder = spheres.Cylinder((0, 0, -1.), (0, 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 1., 1., 0]),
                                                                  np.array([0, -1., 0, 0])))
        nptest.assert_almost_equal(intersection, np.array([0, 1., 0]))

    def test_cylinder_two_intersections_skew(self):
        cylinder = spheres.Cylinder((0, 0, -2.), (0, 0, 2.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 2., 0, 0]),
                                                                  np.array([0, -1., 0, 1.])))
        nptest.assert_almost_equal(intersection, np.array([1., 0, 1.]))

    def test_cylinder_no_intersections_parallel_to_axis(self):
        cylinder = spheres.Cylinder((0, 0, -1.), (0, 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 2.0, 0, 2.]),
                                                                  np.array([0, 0, 0, -1.])))
        assert intersection is None

    def test_cylinder_no_intersections_on_axis(self):
        cylinder = spheres.Cylinder((0, 0, -1.), (0, 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 0, 0, 2.]),
                                                                  np.array([0, 0, 0, -1.])))
        assert intersection is None

    def test_cylinder_not_aligned_on_axis(self):
        cylinder = spheres.Cylinder((-1., 0, -1.), (1., 0, 1.), 1)
        intersection, color = cylinder.get_intersection_and_color(spheres.Ray(np.array([0, 0, 2., 0]),
                                                                  np.array([0, 0., -1., 0])))
        nptest.assert_almost_equal(intersection, np.array([0, 1., 0]))


class TestBoxIntersection:
    def test_box_intersection(self):
        box = spheres.RectangularPrism(2., 2., 2., 0.1)
        intersection, color = box.get_intersection_and_color(spheres.Ray(np.array([0, 2., 1., 0]),
                                                             np.array([0, -1., 0, 0])))
        nptest.assert_almost_equal(intersection, np.array([1.1, 1.0, 0]))


class TestLorentzBoost:
    def test_identity(self):
        beta = np.array([0, 0, 0])
        nptest.assert_almost_equal(util.lorentz_boost(beta),
                                   np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]))

    def test_x_boost(self):
        beta = np.array([0.6, 0, 0])
        nptest.assert_almost_equal(util.lorentz_boost(beta),
                                   np.array([[1.25, -0.75, 0, 0],
                                             [-0.75, 1.25, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]))

    def test_y_boost(self):
        beta = np.array([0, 0.6, 0])
        nptest.assert_almost_equal(util.lorentz_boost(beta),
                                   np.array([[1.25, 0, -0.75, 0],
                                             [0, 1, 0, 0],
                                             [-0.75, 0, 1.25, 0],
                                             [0, 0, 0, 1]]))

    def test_z_boost(self):
        beta = np.array([0, 0, 0.6])
        nptest.assert_almost_equal(util.lorentz_boost(beta),
                                   np.array([[1.25, 0, 0, -0.75],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [-0.75, 0, 0, 1.25]]))

    def test_xyz_boost(self):
        beta = np.array([0.5, 0.5, 0.5])
        nptest.assert_almost_equal(util.lorentz_boost(beta),
                                   np.array([[2, -1, -1, -1],
                                             [-1, 4 / 3., 1 / 3., 1 / 3.],
                                             [-1, 1 / 3., 4 / 3., 1 / 3.],
                                             [-1, 1 / 3., 1 / 3., 4 / 3.]]))
