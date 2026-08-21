Vector and System DE Discovery
==============================

NestyNet_SR supports discovery of coupled multi-equation systems and
vector-valued differential equations, including Maxwell's equations and
Navier-Stokes.

System DE Discovery
-------------------

For coupled systems where each equation may have independent coefficients:

.. math::

   \frac{\partial u_i}{\partial t} = \sum_k c_{ik} \, \phi_k(u, x)

The ``SystemDESearchConfig`` allows independent coefficient matrices per
equation component.

Vector DE Discovery
-------------------

For vector equations with coefficient tying across components:

.. math::

   \frac{\partial \mathbf{E}}{\partial t} = \alpha \, \nabla \times \mathbf{B}

A single shared coefficient vector is used for all components.  This
enforces physical constraints like Maxwell's equations where the same
constant appears in every spatial component.

Vector-System DE Discovery
--------------------------

Multiple coupled vector equations with optional coefficient-sharing groups:

.. math::

   \frac{\partial \mathbf{E}}{\partial t} &= c_1 \, \nabla \times \mathbf{B} \\
   \frac{\partial \mathbf{B}}{\partial t} &= c_2 \, \nabla \times \mathbf{E}

Cross-equation sharing groups enforce that physical constants (e.g.
:math:`c_1 = -c_2` in Maxwell) are discovered consistently.

Vector Calculus Macros
----------------------

The ``vector_ops`` module provides Python-level differential operators
that produce scalar AST nodes:

* ``curl(F, spatial_axes, comps)`` -- :math:`\nabla \times \mathbf{F}`
* ``div(F, spatial_axes, comps)`` -- :math:`\nabla \cdot \mathbf{F}`
* ``grad(f, spatial_axes)`` -- :math:`\nabla f`
* ``laplacian(F, spatial_axes, comps)`` -- :math:`\nabla^2 \mathbf{F}`
* ``advect(v, F, spatial_axes, comps)`` -- :math:`(\mathbf{v} \cdot \nabla) \mathbf{F}`

Usage example::

   from nestynet_sr.sr_de.vector_ops import curl, Vec, VField

   # Build curl terms for Maxwell-like system
   terms = list(curl(E, spatial_axes=(1,2,3), comps=('x','y','z')))

Example: 1D Maxwell
--------------------

Coupled 1D Maxwell equations:

.. math::

   \frac{\partial E}{\partial t} = -\frac{\partial B}{\partial x}, \qquad
   \frac{\partial B}{\partial t} = -\frac{\partial E}{\partial x}

Discovered with cross-coupling coefficients :math:`\approx 1`::

   python tests/test_coupled_1d_maxwell.py

Example: 3D Maxwell
--------------------

Full 3D Maxwell (Ampere + Faraday) with two vector equations sharing
curl terms, coefficient tying, and cross-equation sharing groups::

   python tests/test_maxwell_3d.py

Example: Navier-Stokes
-----------------------

2D Navier-Stokes (Taylor-Green vortex, :math:`\nu = 0.1`) with Laplacian
terms and advection::

   python tests/test_navier_stokes_taylor_green.py
