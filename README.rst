Toy example for inverting geophysical surface magnetic field data
=================================================================

These codes are part of a school project in which the purpose was to
implement a simple statistical inversion algorithms. This was done with
one-dimensional total magnetic field measurements acquired along a linear
profile directed from south to north.

In the model we fit an underground magnetic body with a rectagular
cross-section and infinite extent perpendicular to the measurement
profile. Three inversion parameters are considered of which one has a
linear response and the the two others a nonlinear one. The parameters
are magnetic susceptibility and body's horizontal and vertical
positions.

The inversion algorithm is an iterative one, so that at each step
first-order (i.e. linear) response is estimated with current parameter
values and new parameter values are set by optimizing this linear
model. Eventually the parameter values approach a local optimum and
the iteration converges.

In computing the real responses we have to approximate an integral of
two variables. Originally the integrand was written as Python
function, but this made the code very slow. Much better performance
was obtained by writing the integrand as a Fortran function and
compiling that into a Python extension with ``f2py``.

The Fortran code can be compiled to a Python extension with

::

    f2py -c -m integrand integrand.f

The inversion algorithm is adapted from William Menke's (1988)
"Geophysical data analysis: discrete inverse theory" and the forward
model computation from Richard Blakely's (1995) "Potential theory in
gravity and magnetic applications."
