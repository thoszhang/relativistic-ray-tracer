import numpy as np


def quadratic_eqn_roots(a, b, c):
    """Return roots of ax^2+bx+c *in ascending order*."""

    # TODO: make this more numerically stable

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return []
    elif discriminant == 0:
        return [-b / (2 * a)]
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        return [(-b - sqrt_discriminant) / (2 * a), (-b + sqrt_discriminant) / (2 * a)]


def lorentz_boost(beta):
    """
    Return 4x4 numpy array of Lorentz boost for the velocity 3-vector.

    This is a passive transformation into a reference frame moving at velocity
    = beta with respect to the original frame. Note that c=1.
    """

    beta = np.asarray(beta)

    beta_squared = np.inner(beta, beta)
    if beta_squared >= 1:
        raise ValueError("beta^2 = {} not physically possible".format(beta_squared))
    if beta_squared == 0:
        return np.identity(4)

    gamma = 1 / np.sqrt(1 - beta_squared)

    # see e.g. http://home.thep.lu.se/~malin/LectureNotesFYTA12_2016/SR6.pdf for
    # derivation
    lambda_00 = np.matrix([[gamma]])
    lambda_0j = -gamma * np.matrix(beta)
    lambda_i0 = lambda_0j.transpose()
    lambda_ij = np.identity(3) + (gamma - 1) * np.outer(beta, beta) / beta_squared

    return np.asarray(np.bmat([[lambda_00, lambda_0j], [lambda_i0, lambda_ij]]))


def spherical_angles(point):
    """Return (inclination, azimuth) for the given cartesian coords."""

    x, y, z = point
    radius = np.linalg.norm(point)
    theta = np.arccos(z / radius)
    phi = np.arctan2(y, x) + np.pi
    return theta, phi


def checkerboard(point):
    theta, phi = spherical_angles(point)
    n_theta = int((theta / np.pi) * 12)
    n_phi = int((phi / (2 * np.pi)) * 12)

    return 127 + 128 * ((n_theta + n_phi) % 2)
