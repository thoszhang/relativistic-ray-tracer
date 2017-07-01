#!/usr/bin/env python

import numpy as np
from PIL import Image

ORIGIN = np.array([0, 0, 0, 0])

class Ray(object):
    def __init__(self, start, next_point):
        self.start = start
        self.next_point = next_point

    def boost(self, boost_matrix):
        return Ray(boost_matrix.dot(self.start), boost_matrix.dot(self.next_point))

    def translate(self, offset):
        return Ray(self.start + offset, self.next_point + offset)

class Sphere(object):
    def __init__(self, radius, surface_function, beta, offset):
        self.radius = radius
        self.surface_function = surface_function
        self.beta = beta
        self.offset = offset

    def surface_value(self, x, y, z):
        return self.surface_function(x, y, z)

    def detect_intersection(self, ray):
        x0 = ray.start[1:4]
        x1 = ray.next_point[1:4]
        d = x1 - x0

        a = np.inner(d, d)
        b = 2 * np.inner(x0, d)
        c = np.inner(x0, x0) - self.radius ** 2
        solns = quadratic_eqn_roots(a, b, c)
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
    def __init__(self, image_width, image_height, focal_length, bg_value=0):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.bg_value = bg_value

    def generate_image(self, sphere, time=0, visual_effects=True):
        boost_matrix = lorentz_boost(sphere.beta)
        def image_value(i, j):
            x, y = (j - self.image_width/2, -(i - self.image_height/2))
            return self.trace_ray(x, y, time, sphere, boost_matrix, visual_effects)

        image_matrix = np.fromfunction(np.vectorize(image_value),
                                       (self.image_height, self.image_width))
        return Image.fromarray(image_matrix.astype(np.uint8), mode='L')

    def trace_ray(self, x, y, time, sphere, boost_matrix, visual_effects):
        z = self.focal_length
        if visual_effects:
            origin_to_image_time = spatial_vec_length(x, y, z)
        else:
            origin_to_image_time = 0
        image_coords = np.array([-origin_to_image_time, x, y, z])
        original_ray = Ray(ORIGIN, image_coords).translate(np.array([time, 0, 0, 0]))

        transformed_ray = original_ray.boost(boost_matrix).translate(sphere.offset)

        intersection = sphere.detect_intersection(transformed_ray)
        if intersection is not None:
            return sphere.surface_value(*intersection)
        else:
            return self.bg_value

# see http://home.thep.lu.se/~malin/LectureNotesFYTA12_2016/SR6.pdf
def lorentz_boost(beta):
    beta_squared = np.dot(beta, beta)
    if beta_squared >= 1:
        raise ValueError("beta^2 = {} not physically possible".format(beta_squared))
    if beta_squared == 0:
        return np.identity(4)

    gamma = 1 / np.sqrt(1 - beta_squared)

    lambda_00 = np.matrix([[gamma]])
    lambda_0j = -gamma * np.matrix(beta)
    lambda_i0 = lambda_0j.transpose()
    lambda_ij = np.identity(3) + (gamma-1) * np.outer(beta, beta) / beta_squared

    return np.asarray(np.bmat([[lambda_00, lambda_0j], [lambda_i0, lambda_ij]]))

def quadratic_eqn_roots(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []
    elif discriminant == 0:
        return [-b / (2 * a)]
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        return [(-b - sqrt_discriminant) / (2*a), (-b + sqrt_discriminant) / (2*a)]

def spatial_vec_length(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def spherical_angles(x, y, z):
    radius = spatial_vec_length(x, y, z)
    theta = np.arccos(z / radius)
    phi = np.arctan2(y, x) + np.pi
    return (theta, phi)

def checkerboard(x, y, z):
    theta, phi = spherical_angles(x, y, z)
    n_theta = int((theta / np.pi) * 12)
    n_phi = int((phi / (2*np.pi)) * 12)

    return 127 + 128 * ((n_theta + n_phi) % 2)

def image_sequence():
    height = 150
    width = 600
    focal_length = 200

    beta = np.array([0.5, 0, 0])
    offset = np.array([0, 0, 0, -200])
    radius = 50
    sphere_function = checkerboard

    camera = Camera(width, height, focal_length)
    sphere = Sphere(radius, sphere_function, beta, offset)
    times = [0, 100, 200, 300]
    for time in times:
        image = camera.generate_image(sphere, time, visual_effects=False)
        image.show()
    for time in times:
        image = camera.generate_image(sphere, time, visual_effects=True)
        image.show()

if __name__ == '__main__':
    image_sequence()
