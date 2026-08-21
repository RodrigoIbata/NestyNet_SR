DE Discovery
============

NestyNet_SR discovers differential equations from trajectory data
using sparse regression on a library of candidate terms.

Pipeline
--------

1. **Train surrogate**: Fit a neural network :math:`\hat{u}(x)` or :math:`\hat{u}(x, t)` to the trajectory data
2. **Compute derivatives**: Obtain analytic derivatives :math:`u_x`, :math:`u_{xx}`, etc. from NestyNet
3. **Build library**: Construct a matrix of candidate terms (polynomials, products, derivatives)
4. **STLSQ**: Sequential Thresholded Least Squares for sparse coefficient selection
5. **VarPro Phase 1**: Refine linear coefficients via Variable Projection
6. **VarPro Phase 2**: Search over nonlinear templates (power laws, exponentials)

STLSQ
------

STLSQ iteratively solves the least-squares problem and thresholds small
coefficients:

.. math::

   \dot{u} = \Theta(u, x) \, \xi

where :math:`\Theta` is the library matrix and :math:`\xi` is the sparse
coefficient vector.

At each iteration:

1. Solve :math:`\xi = \Theta^+ \dot{u}` (least squares)
2. Threshold: set :math:`\xi_i = 0` if :math:`|\xi_i| < \lambda`
3. Repeat with the reduced library until convergence

The sparsification threshold :math:`\lambda` (``--stlsq_lambda``) controls
the trade-off between accuracy and parsimony.

VarPro Refinement
-----------------

Phase 1: Linear Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variable Projection analytically optimizes linear coefficients given
the support (active terms) from STLSQ, achieving ~100x improvement
in parameter accuracy.

Phase 2: Nonlinear Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searches over template families to replace polynomial terms with more
compact nonlinear forms:

* **Power laws**: :math:`c \cdot u^p`
* **Exponentials**: :math:`c \cdot e^{\alpha u}`

Initialization uses log-regression heuristics for robust starting points.

Term Library
------------

The default library includes:

* :math:`u, u^2, u^3` -- polynomial terms in the state variable
* :math:`u_x, u_{xx}` -- derivative terms (1st and 2nd order)
* :math:`x \cdot u_x` -- cross terms (for equations like Lane-Emden)
* :math:`u \cdot u_x` -- nonlinear transport terms

Special options:

* ``--include_xdu``: Add :math:`x \cdot u_x` terms (essential for Lane-Emden)
* ``--include_udu``: Add :math:`u \cdot u_x` terms (logistic-like dynamics)

Multi-Dataset Discovery
-----------------------

When multiple experiments share the same DE structure with different
parameters, group-sparse STLSQ discovers the shared support::

   nestynet-de --filepaths data/exp1.csv data/exp2.csv --varpro

The shared support ensures the same terms are active across all datasets,
while per-dataset coefficients capture parameter variations.

Example: Logistic Growth
------------------------

From trajectory data satisfying :math:`\dot{u} = r u (1 - u/K)`::

   nestynet-de --filepath data/logistic.csv --include_udu

Discovers:

.. math::

   \dot{u} = c_1 u + c_2 u^2

with STLSQ correctly identifying the active terms and VarPro refining
coefficients to :math:`< 0.01\%` error.

Example: Lane-Emden Equation
-----------------------------

The Lane-Emden equation:

.. math::

   u_{xx} + \frac{2}{x} u_x + u^n = 0

requires cross terms::

   nestynet-de --filepath data/lane_emden.csv \
       --order_candidates 2 --include_xdu

Configuration Reference
-----------------------

Key CLI options::

   nestynet-de --filepath data.csv \
       --order_candidates 1,2 \
       --include_xdu \
       --stlsq_lambda 0.01 \
       --varpro \
       --varpro_templates power,exp
