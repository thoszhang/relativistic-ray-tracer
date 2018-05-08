#!/usr/bin/env python

import util
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

ORIGIN = np.array([0, 0, 0, 0])
DEFAULT_OBJ_COLOR = 255
DEFAULT_BG_COLOR = 0


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


class RayTracedObject(ABC):
    @abstractmethod
    def get_intersection_and_color(self, ray: Ray):
        """Get the point where the ray intersects with the object, and its color at that point."""


class Sphere(RayTracedObject):
    """A (3d) sphere centered at the origin."""

    def __init__(self, radius, color_function=lambda p: DEFAULT_OBJ_COLOR):
        self.radius = radius
        self.color_function = color_function
        self._radius_sq = radius ** 2

    def get_intersection_and_color(self, ray: Ray):
        intersection = self._get_intersection(ray)
        if intersection is not None:
            return intersection, self.color_function(intersection)
        else:
            return None, None

    def _get_intersection(self, ray: Ray):
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
    def __init__(self, start, end, radius, color=DEFAULT_OBJ_COLOR):
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.radius = radius
        self.color = color
        self._radius_sq = radius ** 2
        self._axis = self.end - self.start
        self._axis_sq = np.inner(self._axis, self._axis)

    def get_intersection_and_color(self, ray: Ray):
        intersection = self._get_intersection(ray)
        if intersection is not None:
            return intersection, self.color
        else:
            return None, None

    def _get_intersection(self, ray: Ray):
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


class CompositeObject(RayTracedObject):
    def __init__(self, objs):
        self.objs = objs

    def get_intersection_and_color(self, ray: Ray):
        x0 = ray.start_3
        min_dist = None
        intersection = None
        color = None
        for obj in self.objs:
            i, c = obj.get_intersection_and_color(ray)
            if i is not None:
                dist = np.linalg.norm(i - x0)
                if not min_dist or dist < min_dist:
                    min_dist = dist
                    intersection = i
                    color = c

        return intersection, color


class RectangularPrism(RayTracedObject):
    """
    A rectangular prism with edges parallel to the coordinate axes, made up of cylinders.
    """

    def __init__(self, width, height, depth, segment_radius, color=DEFAULT_OBJ_COLOR):
        self.width = width
        self.height = height
        self.depth = depth
        self.segment_radius = segment_radius
        self.color = color
        self._cylinders = CompositeObject(self._get_cylinders())

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

    def get_intersection_and_color(self, ray: Ray):
        return self._cylinders.get_intersection_and_color(ray)


class MovingObject:
    def __init__(self, obj: RayTracedObject, beta, offset):
        self.obj = obj
        self.beta = np.asarray(beta)
        self.offset = np.asarray(offset)


class RayTracer:
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

    def __init__(self, image_width, image_height, focal_length, bg_color=DEFAULT_BG_COLOR):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.bg_value = bg_color

    def generate_image(self, moving_object: MovingObject, time):
        boost_matrix = util.lorentz_boost(moving_object.beta)

        def trace_ray(x, y):
            origin_to_image_time = np.linalg.norm([x, y, self.focal_length])
            image_coords = np.array([-origin_to_image_time, x, y, self.focal_length])
            camera_frame_ray = Ray(ORIGIN, image_coords).translate(np.array([time, 0, 0, 0]))

            object_frame_ray = camera_frame_ray.boost(boost_matrix).translate(moving_object.offset)
            intersection, color = moving_object.obj.get_intersection_and_color(object_frame_ray)
            if color:
                return color
            else:
                return self.bg_value

        def image_value(i, j):
            x, y = (j - self.image_width / 2, -(i - self.image_height / 2))
            return trace_ray(x, y)

        image_matrix = np.fromfunction(np.vectorize(image_value),
                                       (self.image_height, self.image_width))
        return Image.fromarray(image_matrix.astype(np.uint8), mode='L')


def image_sequence():
    height = 150
    width = 600
    focal_length = 200

    beta = (0.5, 0, 0)
    offset = (0, 0, 0, -200)

    cube = RectangularPrism(100, 100, 100, 1)
    moving_cube = MovingObject(cube, beta, offset)

    ray_tracer = RayTracer(width, height, focal_length)
    times = [0, 100, 200, 300]
    for time in times:
        image = ray_tracer.generate_image(moving_cube, time)
        image.save("example-{:03d}.png".format(time), "PNG")


if __name__ == '__main__':
    image_sequence()
