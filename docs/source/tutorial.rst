Tutorial: End-to-End Symbolic Regression
=========================================

This walkthrough shows how to generate synthetic data for a known equation,
run the SR pipeline to rediscover it, and inspect the results -- first without
units, then with dimensional analysis enabled.

All commands assume you are in the NestyNet_SR project root directory.

Step 1: Generate Data
---------------------

Use ``data/generate_data.py`` to create a CSV from any expression using SymPy
syntax.  Variables must be named ``x0``, ``x1``, etc.

.. code-block:: bash

   # Example: y = sin(x0 * x1) + x2^2
   python data/generate_data.py \
       --expr "sin(x0 * x1) + x2**2" \
       --output data/tutorial.csv \
       --samples 10000 \
       --min 1.0 --max 5.0

This produces ``data/tutorial.csv`` with columns ``y, x0, x1, x2``.

You can also set per-variable ranges:

.. code-block:: bash

   python data/generate_data.py \
       --expr "sin(x0 * x1) + x2**2" \
       --output data/tutorial.csv \
       --xmin "[0.5, 0.5, 1.0]" --xmax "[3.0, 3.0, 5.0]"

Step 2: Run Symbolic Regression (Without Units)
------------------------------------------------

.. code-block:: bash

   python nestynet_sr/run_SR.py --filepath data/tutorial.csv

This runs the full pipeline:

1. **Stage A** -- trains a neural network surrogate and detects separability
   structure (additive/multiplicative splits, compound variables).
2. **Stage B** -- rewrites surviving neural atoms into analytical forms
   (polynomials, sinusoids, rationals, etc.).

For faster iteration during testing, use ``--fast``:

.. code-block:: bash

   python nestynet_sr/run_SR.py --filepath data/tutorial.csv --fast

Step 3: Inspect the Results
---------------------------

Results are written to ``results/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Content
   * - ``results/tutorial.human``
     - Human-readable discovered expression
   * - ``results/tutorial.state.pkl``
     - Checkpoint (can resume with ``--resume_from``)
   * - ``results/tutorial.report.json``
     - Detailed JSON report with metrics

View the discovered expression::

   cat results/tutorial.human

Step 4: Run with Dimensional Analysis
--------------------------------------

To constrain the search with physical units, provide a unit exponent vector
for *y* and a unit exponent matrix for the input variables.  Each vector lists
exponents in a chosen basis (e.g. Length, Time, Mass).

For the example equation :math:`y = \sin(x_0 \cdot x_1) + x_2^2`, suppose:

* ``x0`` has units of length :math:`[\mathrm{L}]`
* ``x1`` has units of inverse length :math:`[\mathrm{L}^{-1}]`
  (so that :math:`x_0 \cdot x_1` is dimensionless, as required for ``sin``)
* ``x2`` has units of length :math:`[\mathrm{L}]`
* ``y`` has units of length squared :math:`[\mathrm{L}^2]`

.. code-block:: bash

   python nestynet_sr/run_SR.py --filepath data/tutorial.csv \
       --y_units "[2,0,0]" \
       --x_units "[[1,0,0],[-1,0,0],[1,0,0]]" \
       --units_basis "L,T,M"

The three entries in each vector correspond to the exponents of
:math:`[\mathrm{L}, \mathrm{T}, \mathrm{M}]`.
When a units spec is provided, dimensional analysis is enabled by default
(pass ``--ignore_units`` to disable). This activates:

* **Local consistency**: addition requires matching dimensions;
  transcendentals require dimensionless arguments.
* **Global constraint propagation**: all dimensional constraints are solved
  jointly to prune infeasible candidates early.

Step 5: Resume from Checkpoint
------------------------------

If you want to continue a previous run (for instance to extend Stage B):

.. code-block:: bash

   python nestynet_sr/run_SR.py --filepath data/tutorial.csv \
       --resume_from results/tutorial.state.pkl

Command Summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Step
     - Command
   * - Generate data
     - ``python data/generate_data.py --expr "sin(x0*x1)+x2**2" --output data/tutorial.csv``
   * - Run SR (no units)
     - ``python nestynet_sr/run_SR.py --filepath data/tutorial.csv``
   * - Run SR (fast)
     - ``python nestynet_sr/run_SR.py --filepath data/tutorial.csv --fast``
   * - Run SR (with units)
     - ``python nestynet_sr/run_SR.py --filepath data/tutorial.csv --y_units "[2,0,0]" --x_units "[[1,0,0],[-1,0,0],[1,0,0]]"``
   * - View result
     - ``cat results/tutorial.human``
   * - Resume
     - ``python nestynet_sr/run_SR.py --filepath data/tutorial.csv --resume_from results/tutorial.state.pkl``

Next Steps
----------

* :doc:`user_guide/symbolic_regression` -- Detailed SR pipeline guide
* :doc:`user_guide/dimensional_analysis` -- Dimensional analysis and Buckingham-Sudoku
* :doc:`user_guide/de_discovery` -- DE discovery
* :doc:`examples/index` -- More worked examples
