# relativistic-spheres

## Introduction

(In progress)

Objects moving at relativistic speeds are contracted in the direction of motion.
However, the finite speed of light introduces additional visual distortions when an observer looks at an object.
Light rays emitted from points on the moving object that are farther away from the observer take longer to reach the observer, which causes a lengthening of the apparent shape.
This effect is called [Terrell rotation](https://en.wikipedia.org/wiki/Terrell_rotation).

It turns out that the two effects mentioned above, relativistic length contraction and visual distortion, actually cancel each other out.
A sphere moving at relativistic speeds will always have an apparent circular outline.
I learned about this in *Spinors and Spacetime* by Penrose and Rindler, which uses the fact that Lorentz transformations act as conformal transformations on the celestial sphere.
(Here's a [Wikipedia link](https://en.wikipedia.org/wiki/Lorentz_group#Relation_to_the_M.C3.B6bius_group) that outlines that approach.)

I wanted a less abstruse and more direct way of visualizing this fact, so I wrote a simple "ray tracer" that models the light rays directly.
A sphere in this model has a velocity and a direction with respect to the camera.
Like an ordinary ray tracer, this program models rays of light originating from the observer and calculates where they hit the sphere.
However, the light rays in this program are actually rays in 4-dimensional Minkowski space.
Each ray has a Lorentz transformation applied to it to get it into the sphere's reference frame, and then the usual sphere collision calculation is performed.
(There's no shading or recursion in this model.)

A lot of these ideas were taken from [_Relativistic Ray-Tracing: Simulating the Visual Appearance of Rapidly Moving Objects_](https://www.researchgate.net/publication/2574193_Relativistic_Ray-Tracing_Simulating_the_Visual_Appearance_of_Rapidly_Moving_Objects) by Howard, Dance, and Kitchen.

## How to use

Dependencies are `numpy` and `pillow`.

The `RayTracer` class models a "camera" with a specified image width, image height, and focal length. The `Sphere` class models a sphere moving with a certain velocity and at a certain offset with respect to the camera. All 4-vectors (and some 3-vectors; it's not that consistent) are numpy arrays.


