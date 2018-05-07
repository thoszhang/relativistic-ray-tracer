#!/usr/bin/env python

import util
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from PIL import Image

ORIGIN = np.array([0, 0, 0, 0])

MovingObject = namedtuple('MovingObject', ['obj', 'beta', 'offset'])


class RayTracedObject(ABC):
    @abstractmethod
    def color(self, ray):
        """Get the color of the object where it intersects with a ray."""

    @abstractmethod
    def intersection(self, ray):
        """Get the intersection point of a ray with the object."""


class Ray:
    """A ray in Minkowski space."""

    def __init__(self, start, direction):
        self.start = start
        self.direction = direction
        self.start_3 = start[1:4]
        self.direction_3 = direction[1:4]

    def boost(self, boost_matrix):
        return Ray(boost_matrix.dot(self.start), boost_matrix.dot(self.direction))

    def translate(self, offset):
        return Ray(self.start + offset, self.direction)


class Sphere(RayTracedObject):
    """A (3d) sphere centered at the origin."""

    def __init__(self, radius, color_function=None):
        self.radius = radius
        self.color_function = color_function
        self._radius_sq = radius ** 2

    def color(self, ray):
        point = self.intersection(ray)
        if point is not None:
            return self.color_function(point)
        else:
            return None

    def intersection(self, ray):
        x0 = ray.start_3
        d = ray.direction_3

        a = np.inner(d, d)
        b = 2 * np.inner(x0, d)
        c = np.inner(x0, x0) - self._radius_sq
        solns = util.quadratic_eqn_roots(a, b, c)
        for root in solns:
            if root >= 0:
                return x0 + root * d

        return None


class Cylinder(RayTracedObject):
    def __init__(self, start, end, radius, color):
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.radius = radius
        self.color = color
        self._radius_sq = radius ** 2
        self._axis = self.end - self.start
        self._axis_sq = np.inner(self._axis, self._axis)

    def color(self, ray):
        point = self.intersection(ray)
        if point is not None:
            return self.color
        else:
            return None

    def intersection(self, ray):
        # TODO: document this better
        x0 = ray.start_3
        d = ray.direction_3
        d_proj = d - (np.inner(d, self._axis) / self._axis_sq) * self._axis

        q = x0 - self.start
        q_proj = q - (np.inner(q, self._axis) / self._axis_sq) * self._axis

        a = np.inner(d_proj, d_proj)
        if a == 0:
            return None
        b = 2 * np.inner(d_proj, q_proj)
        c = np.inner(q_proj, q_proj) - self._radius_sq

        solns = util.quadratic_eqn_roots(a, b, c)
        for root in solns:
            if root >= 0:
                x = x0 + root * d
                # parameter for the cylinder axis line segment
                s = np.inner(x - self.start, self._axis) / self._axis_sq
                if 0 <= s <= 1:
                    return x

        return None


class RectangularPrism(RayTracedObject):
    """
    A box with a width, length, and height, made up of cylinders.
    """

    def __init__(self, width, height, depth, segment_radius, color):
        self.width = width
        self.height = height
        self.depth = depth
        self.segment_radius = segment_radius
        self.color = color
        self._cylinders = MultipleObjects(self._get_cylinders())

    def _get_cylinders(self):
        x = self.width / 2.0 + self.segment_radius
        y = self.height / 2.0 + self.segment_radius
        z = self.depth / 2.0 + self.segment_radius

        endpoints = [
            # "front" rectangle
            ((+x, +y, +z), (+x, -y, +z)),
            ((+x, -y, +z), (-x, -y, +z)),
            ((-x, -y, +z), (-x, +y, +z)),
            ((-x, +y, +z), (+x, +y, +z)),
            # "back" rectangle
            ((+x, +y, -z), (+x, -y, -z)),
            ((+x, -y, -z), (-x, -y, -z)),
            ((-x, -y, -z), (-x, +y, -z)),
            ((-x, +y, -z), (+x, +y, -z)),
            # connect the rectangles to make a prism
            ((+x, +y, +z), (+x, +y, -z)),
            ((+x, -y, +z), (+x, -y, -z)),
            ((-x, -y, +z), (-x, -y, -z)),
            ((-x, +y, +z), (-x, +y, -z)),
        ]
        return [Cylinder(start, end, self.segment_radius, self.color) for start, end in endpoints]

    def color(self, ray):
        return self._cylinders.color(ray)

    def intersection(self, ray):
        return self._cylinders.intersection(ray)


class MultipleObjects(RayTracedObject):
    def __init__(self, objs):
        self.objs = objs

    def color(self, ray):
        closest_obj, closest_point = self._get_closest_intersecting_object(ray)
        if closest_obj is not None:
            return closest_obj.color(closest_point)
        else:
            return None

    def intersection(self, ray):
        closest_obj, closest_point = self._get_closest_intersecting_object(ray)
        return closest_point

    def _get_closest_intersecting_object(self, ray):
        x0 = ray.start_3
        min_dist = None
        closest_point = None
        closest_obj = None
        for obj in self.objs:
            point = obj.detect_intersection(ray)
            if point is not None:
                dist = np.linalg.norm(point - x0)
                if not min_dist or dist < min_dist:
                    min_dist = dist
                    closest_point = point
                    closest_obj = obj

        return closest_obj, closest_point


class Camera:
    """
    An ideal pinhole camera.

    The pinhole of the camera is at the (spatial) origin, and it faces the +z
    direction. Incoming light rays enter through the pinhole and strike a flat
    screen at z = -`focal_length`. For a given picture, all of the light rays
    enter through the pinhole at the same time (even though light rays at the
    edges of the picture would have struck the screen later).

    The image must then be flipped (as with all pinhole cameras) to produce the
    correctly oriented image; that is, an ray that hit the screen at (-x, -y,
    -z) corresponds to the point (x, y) in the final image.

           +z
        \   |   /
         v  v  v  direction of light
          \ | /
           \|/
            *  pinhole (z = 0)
           /|\
          v v v
        _/__|__\_  screen
           -z

    If a ray hit the screen at (-x, -y, -z), then we can find the point on the
    sphere that emitted the ray using backward ray tracing. The backward ray
    starts at the pinhole at (0, 0, 0), and a point on the ray is (x, y, z).
    Considered as points in Minkowski space, the light entered the pinhole at
    (`time`, 0, 0, 0), and the point on the ray is at (`time`-t1, x, y, z),
    where t1 is the amount of time taken for the original (non-backward) ray to
    go from (x, y, z) to (0, 0, 0).

    These ideas were taken from  "Relativistic Ray-Tracing: Simulating the
    Visual Appearance of Rapidly Moving Objects" (1995) by Howard, Dance, and
    Kitchen.

    Note also that because light rays are projected onto a flat screen, there
    will be distortion around the edges of the image, since an object that
    subtends a certain angle will be projected onto a larger surface area when
    its position vector makes a larger angle with respect to the z axis.
    """

    def __init__(self, image_width, image_height, focal_length, bg_value=0):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.bg_value = bg_value

    def generate_image(self, scene_object, time, beta, offset):
        boost_matrix = util.lorentz_boost(beta)

        def image_value(i, j):
            x, y = (j - self.image_width / 2, -(i - self.image_height / 2))
            return self.detect_intersection(x, y, time, scene_object, offset, boost_matrix)

        image_matrix = np.fromfunction(np.vectorize(image_value),
                                       (self.image_height, self.image_width))
        return Image.fromarray(image_matrix.astype(np.uint8), mode='L')

    def detect_intersection(self, x, y, time, scene_object, offset, boost_matrix):
        z = self.focal_length
        origin_to_image_time = util.spatial_vec_length(x, y, z)
        image_coords = np.array([-origin_to_image_time, x, y, z])
        original_ray = Ray(ORIGIN, image_coords).translate(np.array([time, 0, 0, 0]))

        transformed_ray = original_ray.boost(boost_matrix).translate(offset)
        intersection = scene_object.detect_intersection(transformed_ray)
        if intersection is not None:
            return 255
        else:
            return self.bg_value

    def trace_ray(self, x, y, time, sphere, boost_matrix, visual_effects):
        z = self.focal_length
        if visual_effects:
            origin_to_image_time = util.spatial_vec_length(x, y, z)
        else:
            # assume infinite speed of light only for the light rays from the
            # object to the camera
            origin_to_image_time = 0
        image_coords = np.array([-origin_to_image_time, x, y, z])
        original_ray = Ray(ORIGIN, image_coords).translate(np.array([time, 0, 0, 0]))

        transformed_ray = original_ray.boost(boost_matrix).translate(sphere.offset)

        intersection = sphere.detect_intersection(transformed_ray)
        if intersection is not None:
            return sphere.surface_value(*intersection)
        else:
            return self.bg_value


def image_sequence():
    height = 150
    width = 600
    focal_length = 200

    beta = np.array([0.5, 0, 0])
    offset = np.array([0, 0, 0, -200])

    cube = RectangularPrism(100, 100, 100, 1)


    # radius = 50
    # sphere_function = util.checkerboard

    camera = Camera(width, height, focal_length)
    # sphere = Sphere(radius, sphere_function, beta, offset)
    times = [0, 100, 200, 300]
    for time in times:
        image = camera.generate_image(cube, time, beta, offset)
        image.save("example-{:03d}-visual-effects-off.png".format(time), "PNG")


if __name__ == '__main__':
    image_sequence()
