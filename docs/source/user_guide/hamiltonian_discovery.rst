Hamiltonian Discovery
=====================

NestyNet_SR discovers Hamiltonian functions :math:`H(q, p)` from phase-space
trajectories, automatically preserving symplectic structure.

Approach
--------

Given phase-space data :math:`(q(t), p(t))`, the framework:

1. Builds a polynomial library in :math:`q` and :math:`p`
2. Enforces the mechanical split: :math:`H = T(p) + V(q)` (kinetic + potential)
3. Applies STLSQ to discover active terms
4. Optionally refines via autograd-based design matrix

Hamilton's equations provide the constraint:

.. math::

   \dot{q} = \frac{\partial H}{\partial p}, \qquad
   \dot{p} = -\frac{\partial H}{\partial q}

Discovery Modes
---------------

Mode A: Fully Shared H
~~~~~~~~~~~~~~~~~~~~~~~

A single Hamiltonian is shared across all datasets.  All datasets contribute
to the same coefficient discovery::

   python tests/hamiltonian/test_hamiltonian_mode_a.py

Mode B: Group-STLSQ
~~~~~~~~~~~~~~~~~~~~

Shared support (same active terms) but per-dataset coefficients.  Useful
when multiple systems share the same functional form but with different
physical parameters::

   python tests/hamiltonian/test_hamiltonian_multi.py

Contact Hamiltonian Extension
-----------------------------

For damped systems, the framework supports contact Hamiltonian dynamics:

.. math::

   \dot{q} = \frac{\partial H_0}{\partial p}, \qquad
   \dot{p} = -\frac{\partial H_0}{\partial q} - \gamma p

where :math:`\gamma` is the damping coefficient.

Dimensional Analysis
--------------------

Hamiltonian discovery supports dimensional analysis to ensure physically
consistent terms.  Dimensional analysis is enabled by default when a units
spec is provided (pass ``--ignore_units`` to disable)::

   python tests/hamiltonian/test_hamiltonian_units.py

Examples
--------

Simple Harmonic Oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Discover :math:`H = \frac{p^2}{2m} + \frac{k q^2}{2}` from
:math:`(q(t), p(t))` trajectories::

   python tests/hamiltonian/test_hamiltonian_sho.py

Coefficients converge to :math:`\approx 0.5` for both :math:`p^2` and :math:`q^2`.

Pendulum
~~~~~~~~

Discover :math:`H = p^2/2 - \cos(q)` from pendulum phase portraits.
