User Guide
==========

This guide provides detailed information about using NestyNet_SR for symbolic
regression and differential equation discovery.

.. toctree::
   :maxdepth: 2

   symbolic_regression
   factorized_search_closure_machine
   de_discovery
   vector_de
   hamiltonian_discovery
   poisson_geometry
   dimensional_analysis
   generalized_symmetries
   nonlinear_de_symmetries

Overview
--------

NestyNet_SR discovers analytical expressions and governing equations from data
through a two-stage pipeline:

1. **Stage A** -- Train a neural network surrogate, detect separability structure
2. **Stage B** -- Rewrite neural atoms into analytical forms

The framework supports:

* **Symbolic regression**: :math:`y = f(x)` from tabular data
* **DE discovery**: :math:`u' = F(u, x)` from trajectory data
* **System DE discovery**: coupled multi-equation systems
* **Vector DE discovery**: vector equations with coefficient tying (e.g. Maxwell's equations)
* **Hamiltonian discovery**: :math:`H(q, p)` from phase-space trajectories

All discovery modes share the same underlying NestyNet surrogate architecture,
benefiting from accurate analytic derivatives (10--100x better than autograd).
