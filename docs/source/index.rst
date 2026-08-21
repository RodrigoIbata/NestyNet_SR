.. NestyNet_SR documentation master file

NestyNet\_SR Documentation
==========================

**NestyNet_SR** is a symbolic regression and differential equation discovery
package built on top of NestyNet.  It uses neural network surrogates with
accurate analytic derivatives to discover analytical expressions :math:`y = f(x)`
and governing equations from data.

Key Features
------------

* **Two-Stage Pipeline**: (A) Train neural surrogate and detect separability structure; (B) rewrite neural atoms into analytical forms
* **Separability Detection**: Additive, multiplicative, and compound-variable structure discovery via mixed partial derivatives
* **Stage B Rewriting**: 40+ specialized rules converting neural atoms to polynomials, sinusoids, rationals, and more
* **DE Discovery**: STLSQ sparse regression with VarPro refinement for differential equation identification
* **Vector/System DE**: Coupled multi-equation discovery with vector-calculus macros and coefficient tying
* **Hamiltonian Discovery**: Phase-space trajectory analysis preserving symplectic structure
* **Dimensional Analysis**: Local consistency checks plus global constraint propagation (Buckingham-Sudoku)
* **factorized symbolic search**: The typed closure machine -- brute-enumerate and UCB-guided mutation engine for expression tree search
* **Generalized Symmetries**: Affine and bounded-quadratic point-symmetry discovery with determining certificates, invariant compilation, and symmetry reduction
* **Poisson/Casimir Discovery**: General Poisson geometry, Darboux charts, Noether charges, and certified Casimir invariants of recovered symmetry algebras
* **Statistical Selection**: Archive-conditional certification with frozen candidate archives and simultaneous confidence Pareto fronts

Quick Start
-----------

Installation::

   # Install NestyNet first (required dependency)
   cd /path/to/NestyNet
   pip install -e .

   # Then install NestyNet_SR
   cd /path/to/NestyNet_SR
   pip install -e .

Basic symbolic regression::

   nestynet-sr --filepath data/my_data.csv

DE discovery::

   nestynet-de --filepath data/my_ode_data.csv

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorial
   user_guide/index
   api/index
   statistical_selection
   examples/index
   development

Additional Documentation
------------------------

The design note for the archive-conditional statistical model selection layer
(frozen candidate archives, common-domain loss audits, and simultaneous
confidence Pareto fronts) is maintained as a standalone Markdown document at
``docs/statistical_selection.md`` in the source tree; the corresponding API
reference is :doc:`api/stat_selection`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
