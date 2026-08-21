Getting Started
===============

This guide will help you get started with NestyNet_SR.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

NestyNet_SR requires:

* Python >= 3.10
* PyTorch >= 2.6
* NumPy >= 2.2
* NestyNet (the underlying neural network library)

Install from Source
~~~~~~~~~~~~~~~~~~~

NestyNet must be installed first::

   cd /path/to/NestyNet
   pip install -e .

Then install NestyNet_SR::

   cd /path/to/NestyNet_SR
   pip install -e .

Optional extras::

   pip install -e ".[gui]"   # Streamlit GUI
   pip install -e ".[dev]"   # Development tools (pytest, ruff, mypy)

Command-Line Tools
------------------

Symbolic Regression
~~~~~~~~~~~~~~~~~~~

Discover an analytical expression :math:`y = f(x)` from a CSV file::

   nestynet-sr --filepath data/my_data.csv

Key options:

* ``--fast``: Reduced epochs for quick testing
* ``--no_stageB``: Stage A only (skip analytical rewrites)
* ``--ignore_units``: Disable dimensional analysis (enabled by default when units are provided)
* ``--resume_from results/<stem>.state.pkl``: Resume from checkpoint

DE Discovery
~~~~~~~~~~~~

Discover a differential equation from trajectory data::

   nestynet-de --filepath data/my_ode_data.csv

Key options:

* ``--order_candidates 1,2``: Try both 1st and 2nd order DEs
* ``--include_xdu``: Include :math:`x \cdot u_x` terms (e.g. Lane-Emden)
* ``--include_udu``: Include :math:`u \cdot u_x` terms (e.g. logistic)
* ``--varpro``: Enable Variable Projection refinement
* ``--varpro_templates power,exp``: Template families for Phase 2

Multi-Dataset DE Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discover shared DE structure across multiple experiments::

   nestynet-de --filepaths data/exp1.csv data/exp2.csv --varpro

Streamlit GUI
~~~~~~~~~~~~~

Launch the interactive GUI::

   streamlit run nestynet_sr/gui_sr.py

Pipeline Overview
-----------------

Stage A: Surrogate Training and Structure Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Train a segmented neural network surrogate :math:`\hat{f}(x)` using NestyNet
2. Analyse mixed partial derivatives to detect separability:

   * **Additive**: :math:`f(x,y) = g(x) + h(y)` (detected via :math:`\partial^2 f / \partial x \partial y \approx 0`)
   * **Multiplicative**: :math:`f(x,y) = g(x) \cdot h(y)` (detected via :math:`\partial^2 \log|f| / \partial x \partial y \approx 0`)
   * **Compound variables**: :math:`f(x,y) = g(x^a y^b)` (detected via rank-1 structure in log-space)

3. Build an AST (Abstract Syntax Tree) with neural atoms at the leaves

Stage B: Analytical Rewriting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run factorized symbolic search explorer on each neural atom (brute enumeration + UCB mutations)
2. Apply 40+ specialized rewrite rules: polynomials, sinusoids, rationals, power laws, exponentials
3. Accept rewrites that improve or maintain fit quality (Occam's razor)
4. Iterate until all atoms are analytical or no further progress

Output Files
~~~~~~~~~~~~

Results are written to ``results/``:

* ``<stem>.human``: Human-readable expression
* ``<stem>.state.pkl``: Serialized state for resumption
* ``<stem>.report.json``: Structured report with metrics

Next Steps
----------

* :doc:`user_guide/symbolic_regression` -- Detailed SR pipeline guide
* :doc:`user_guide/de_discovery` -- DE discovery
* :doc:`user_guide/dimensional_analysis` -- Dimensional analysis and Buckingham-Sudoku
* :doc:`examples/index` -- Complete worked examples
