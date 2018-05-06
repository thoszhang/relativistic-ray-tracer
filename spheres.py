#!/usr/bin/env python

import util
import numpy as np
from PIL import Image

ORIGIN = np.array([0, 0, 0, 0])


class Ray(object):
    """A ray in Minkowski space."""

    def __init__(self, start, direction):
        self.start = start
        self.direction = direction

    def boost(self, boost_matrix):
        return Ray(boost_matrix.dot(self.start), boost_matrix.dot(self.direction))

    def translate(self, offset):
        return Ray(self.start + offset, self.direction)


class Sphere(object):
    """
    A 3d sphere with a radius.

    Note that `radius` and `surface_function` are intrinsic to the sphere, but
    `beta` and `offset` are given with respect to the camera and in the camera's
    reference frame.
    """

    def __init__(self, radius, surface_function, beta, offset):
        self.radius = radius
        self.surface_function = surface_function
        self.beta = beta
        self.offset = offset

    def surface_value(self, x, y, z):
        return self.surface_function(x, y, z)

    def detect_intersection(self, ray):
        """
        Return the intersection of a ray with this sphere that is closest to
        the ray's start point.
        """

        x0 = ray.start[1:4]
        d = ray.direction[1:4]

        a = np.inner(d, d)
        b = 2 * np.inner(x0, d)
        c = np.inner(x0, x0) - self.radius ** 2
        solns = util.quadratic_eqn_roots(a, b, c)
        t = None
        for root in solns:
            if root >= 0:
                t = root
                break

        if t:
            return x0 + t * d
        else:
            return None


class Camera(object):
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

    def generate_image(self, sphere, time=0, visual_effects=True):
        """
        Generate a "ray-traced" image of a sphere moving at some velocity.

        `time` is the time at which light rays enter the pinhole.
        `visual_effects` can be turned off in order to see the actual dimensions
        of the moving sphere, i.e., with length contraction, without the visual
        effects.
        """

        boost_matrix = util.lorentz_boost(sphere.beta)

        def image_value(i, j):
            x, y = (j - self.image_width / 2, -(i - self.image_height / 2))
            return self.trace_ray(x, y, time, sphere, boost_matrix, visual_effects)

        image_matrix = np.fromfunction(np.vectorize(image_value),
                                       (self.image_height, self.image_width))
        return Image.fromarray(image_matrix.astype(np.uint8), mode='L')

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
    radius = 50
    sphere_function = util.checkerboard

    camera = Camera(width, height, focal_length)
    sphere = Sphere(radius, sphere_function, beta, offset)
    times = [0, 100, 200, 300]
    for time in times:
        image = camera.generate_image(sphere, time, visual_effects=False)
        image.save("example-{:03d}-visual-effects-off.png".format(time), "PNG")
    for time in times:
        image = camera.generate_image(sphere, time, visual_effects=True)
        image.save("example-{:03d}-visual-effects-on.png".format(time), "PNG")


if __name__ == '__main__':
    image_sequence()
