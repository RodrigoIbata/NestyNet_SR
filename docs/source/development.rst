Development Guide
=================

This guide is for developers contributing to NestyNet_SR.

Development Setup
-----------------

Install with development dependencies::

   cd /path/to/NestyNet
   pip install -e .

   cd /path/to/NestyNet_SR
   pip install -e ".[dev]"

Code Organization
-----------------

Project Structure
~~~~~~~~~~~~~~~~~

::

   nestynet_sr/
   ├── sr_core/           # Core AST nodes, units, separability
   │   ├── bridges.py     # AST node types (Node union)
   │   ├── units.py       # Dimensional analysis (Level 1 + Level 2)
   │   ├── separability_math.py  # Separability detection algorithms
   │   ├── atoms.py       # Atom type definitions
   │   ├── sympy_bridge.py       # SymPy <-> NestyNet AST conversion
   │   └── problem_dims.py       # Canonical benchmark-family metadata
   ├── sr_search/         # Search orchestration
   │   ├── search.py      # Main search loop facade
   │   ├── _search_*.py   # Search loop split (compounds, detection, policy,
   │   │                  #   proposals, runtime, shadow, structure, training)
   │   ├── stageB/        # Stage B rewrite engine
   │   │   ├── rules.py   # Rule core (40+ rules total)
   │   │   ├── rules_*.py # Rule split modules (common, compound,
   │   │   │              #   gauge_homogeneity, nn_leaf, phase_trig,
   │   │   │              #   preconditioner, problem, univariate)
   │   │   ├── engine.py  # Rule dispatcher
   │   │   ├── _engine_runtime.py / _engine_state.py / _engine_support.py
   │   │   └── main.py    # Pattern matchers
   │   ├── factorized_search/       # factorized symbolic search explorer
   │   │   ├── config.py  # FactorizedSearchConfig
   │   │   └── bridge.py  # Tuple-AST <-> Node-AST bridge
   │   ├── candidate_builders.py  # Compound proposals facade
   │   ├── _candidate_builders_*.py  # Compound proposals split (common,
   │   │                             #   multivariate, structural, univariate)
   │   └── compound_functions.py  # Compound function macros
   ├── sr_de/             # Differential equation discovery
   │   ├── de_search.py   # DE discovery pipeline
   │   ├── system_de_search.py  # System/vector DE
   │   ├── vector_ops.py  # Vector calculus macros
   │   ├── de_templates.py      # VarPro templates
   │   ├── varpro_de.py   # VarPro coefficient refinement
   │   ├── complex_ops.py # Complex-field DE discovery
   │   ├── hamiltonian_search.py  # Hamiltonian discovery
   │   ├── poisson_*.py   # General Poisson geometry
   │   ├── factorized_de.py     # Factorized DE rescue facade
   │   └── _factorized_de_*.py  # Factorized DE split (explorer, frontend,
   │                            #   lanes, operator, rescue, search)
   ├── sr_gs/             # Generalized-symmetry layer
   │   ├── affine_algebra.py    # Affine determining operator
   │   ├── nonlinear_de_symmetry.py  # Bounded quadratic point symmetries
   │   ├── de_invariant_compiler.py  # Symbolic invariant compiler
   │   ├── de_reduction.py      # Canonical-coordinate reduction + cascade
   │   ├── noether_reduction.py # Noether charges / momentum maps
   │   └── algebra_casimirs.py  # Certified Casimir discovery
   ├── stat_selection/    # Archive-conditional statistical selection
   │   ├── archive.py     # Frozen candidate archives
   │   ├── pareto.py      # Confidence Pareto fronts
   │   └── sr_pipeline.py / de_pipeline.py  # SR / DE adapters
   ├── discovery/         # Active learning + closed-loop discovery
   │   ├── integration.py # Orchestration
   │   ├── committee.py   # Committee ensembles
   │   └── active_design.py     # Experiment selection
   ├── sr_expr_ir/        # Opt-in quotient-DAG expression IR
   │   ├── qdag.py        # QDAG canonicalization
   │   └── egraph_normalizer.py # Bounded e-graph facade
   ├── adaptors/          # NestyNet optimization adaptors
   │   └── ast_composite.py     # ASTComposite adaptor
   ├── run_SR.py          # CLI entry point (nestynet-sr)
   └── run_de.py          # CLI entry point (nestynet-de)

   tests/                 # Test scripts
   examples/              # Worked examples

Testing
-------

Run Tests
~~~~~~~~~

Tests are standalone scripts in ``tests/`` and ``examples/``::

   # Individual tests
   python tests/test_vector_hello_world.py
   python tests/test_wave_equation.py
   python tests/test_maxwell_3d.py
   python tests/hamiltonian/test_hamiltonian_sho.py

   # Adjoint symmetry tests
   python tests/test_ast_composite_jvp_vjp_adjoint_audit.py

Linting
-------

::

   # Run ruff linter
   ruff check nestynet_sr/

   # Auto-fix (caution: may break re-export hubs)
   ruff check --fix nestynet_sr/

   # Format
   ruff format nestynet_sr/

   # Type check
   mypy nestynet_sr/

.. warning::

   Running ``ruff check --fix`` can remove "unused" imports from re-export hub
   files (``stageB/helpers.py``, ``stageB/__init__.py``).  Always audit these
   files after running auto-fix.  Re-export imports should be tagged with
   ``# noqa: F401``.

Key Conventions
---------------

* **float64 by default**: All computations use double precision for scientific accuracy
* **Exact rational arithmetic**: Dimensional analysis uses ``fractions.Fraction`` throughout
* **AST immutability**: AST nodes should be treated as immutable; use ``clone_ast()`` for modifications
* **Re-export hubs**: ``stageB/helpers.py`` and ``stageB/__init__.py`` are pure re-export files; keep their ``__all__`` lists in sync with imports

Adding a New Stage B Rewrite Rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add the rule function in ``nestynet_sr/sr_search/stageB/rules.py``
2. Register it in ``stageB/engine.py``
3. Add the pattern matcher in ``stageB/main.py``

Adding a New DE Template Family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Define template class in ``nestynet_sr/sr_de/de_templates.py``
2. Implement initialization and parameter bounds
3. Register in the template factory
4. Add to ``--varpro_templates`` choices in ``run_de.py``

Adding Vector Calculus Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add operator function in ``nestynet_sr/sr_de/vector_ops.py`` returning scalar AST nodes
2. Use in system DE discovery via ``extra_terms`` or ``vector_terms`` parameters
