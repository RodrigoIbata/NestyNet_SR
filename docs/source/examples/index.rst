Examples
========

Worked examples live in ``examples/`` (each with its own README and entry
points), and a number of test scripts double as compact worked examples.
See ``examples/README.md`` for per-example quick-start commands.

Symbolic Regression
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/classSR/``
     - Class-SR smoke runs through ``run_SR.py``: shared symbolic form across
       related datasets (damped springs, quadratic families) with
       dataset-specific coefficients.
       Entry points: ``smoke_class_sr.py``, ``smoke_quadratic_class.py``
   * - ``examples/sparc_carrier/``
     - SPARC baryonic-acceleration-carrier vignette (Paper III): blind
       discovery of the carrier :math:`z = g_{\mathrm{gas}} + \Upsilon_d\,
       g_{\mathrm{disk}}` from real bulgeless-galaxy rotation curves.
       Entry points: ``build_dataset.py``, ``run_pilot.py``
   * - ``examples/oracle_factorized_search/``
     - Surrogate-free oracle harness for factorized symbolic search and
       continuous skeleton refinement, with CLI and Streamlit GUIs
       (equation and DE modes).
       Entry points: ``oracle_lab.py``, ``oracle_lab_streamlit.py``,
       ``oracle_lab_de_streamlit.py``

Scalar DE Discovery
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/logistic_growth/``
     - Logistic growth ODE :math:`du/dt = r u (1 - u/K)`: first-order
       discovery plus template optimization.
       Entry point: ``smoke_logistic_discovery.py``
   * - ``examples/lane_emden/``
     - Lane-Emden equation
       :math:`y'' + (2/x)\,y' + y^n = 0`: second-order discovery with a
       singular term.
       Entry point: ``smoke_lane_emden_discovery.py``
   * - ``examples/dho/``
     - Damped harmonic oscillator
       :math:`y'' + \gamma y' + \omega^2 y = 0` from raw trajectory data,
       both direct DE and SR-first DE routes.
       Entry points: ``smoke_dho_discovery.py``, ``smoke_dho_discovery_sr.py``
   * - ``examples/multi_dataset/``
     - Multi-dataset ODE discovery with shared term support (logistic family
       with varying :math:`r`).
       Entry point: ``smoke_multi_logistic.py``
   * - ``examples/feynman_de/``
     - Scalar DE benchmark used by Paper IV: 57 first/second-order ODEs from
       physics (exponential decay, Lane-Emden, Bessel, driven/damped
       oscillators, ...), multi-trajectory engines, declared-class
       ``singular_origin`` metadata keeping the term library answer-blind.
       Entry point: ``run_benchmark.py``
   * - ``examples/feynman_de_coe/``
     - Repeatable detached launch scripts for overnight DE
       Committee-of-Experts validation runs over the scalar DE control cases
       (wraps ``scripts/run_feynman_de_coe_control_suite.py``).
       Entry point: ``launch_full_adjudicate_detached.sh``
   * - ``examples/MOND/``
     - Nonlinear modified-Poisson benchmark
       :math:`\nabla \cdot \left(\mu(|\nabla\phi|/a_0)\, \nabla\phi\right)
       = 4\pi G \rho`.
       Entry point: ``run_benchmark.py``

Complex-Valued DE Discovery
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/feynman_complex/``
     - 26-problem complex-valued DE benchmark (Schrödinger, NLS,
       Ginzburg-Landau, Dirac, Klein-Gordon, ...): ODEs and PDEs via real
       decomposition :math:`\psi = u + iv` and factorized symbolic search
       over coupled-real feature tables.
       Entry point: ``run_benchmark.py``

Vector and System DEs
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/Maxwell/``
     - Coupled vector PDE discovery:
       :math:`\partial\mathbf{E}/\partial t = \nabla \times \mathbf{B}`,
       :math:`\partial\mathbf{B}/\partial t = -\nabla \times \mathbf{E}`,
       plus source and conductive variants.
       Entry points: ``discover_maxwell_*.py``
   * - ``tests/test_vector_hello_world.py``
     - Order-0 algebraic vector system
   * - ``tests/test_wave_equation.py``
     - 1D wave equation :math:`u_{tt} = c^2 u_{xx}`
   * - ``tests/test_vector_curl_equation.py``
     - Vector curl equation :math:`\partial\mathbf{E}/\partial t = \nabla \times \mathbf{B}`
   * - ``tests/test_coupled_1d_maxwell.py``
     - Coupled 1D Maxwell (cross-derivative)
   * - ``tests/test_maxwell_3d.py``
     - Full 3D Maxwell (Ampere + Faraday)
   * - ``tests/test_navier_stokes_taylor_green.py``
     - 2D Navier-Stokes Taylor-Green vortex

Hamiltonian and Geometric Mechanics
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/hamiltonian/``
     - Hamiltonian discovery from phase-space trajectories:
       :math:`H = p^2/2 + q^2/2 + q^4/4`.
       Entry point: ``anharmonic_oscillator.py``
   * - ``examples/poisson_geometry/``
     - General polynomial Poisson-bracket discovery with shared Hamiltonian
       heads, plus a Casimir taxonomy (physical Poisson Casimir vs full-rank
       algebra invariant, Hamiltonian gauge quotient); quadratic cyclic
       Lotka-Volterra and affine translated-Euler brackets.
       Entry points: ``cyclic_lotka_volterra.py``, ``translated_euler_top.py``,
       ``casimir_taxonomy.py``
   * - ``tests/hamiltonian/test_hamiltonian_sho.py``
     - Simple harmonic oscillator :math:`H = \tfrac{1}{2}q^2 + \tfrac{1}{2}p^2`
   * - ``tests/hamiltonian/test_hamiltonian_const_term.py``
     - Constant term handling
   * - ``tests/hamiltonian/test_hamiltonian_multi.py``
     - Multi-dataset with shared support
   * - ``tests/hamiltonian/test_hamiltonian_units.py``
     - Dimensional analysis consistency
   * - ``tests/hamiltonian/test_hamiltonian_mode_a.py``
     - Mode A (fully shared H) vs Mode B (group-STLSQ)

Generalized Symmetries and Charts
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/generalized_symmetries/``
     - Entry points for the generalized-symmetry (GS) layer: analytic
       affine-generator audit and the Stage-A/DE smoke benchmark
       (e.g. :math:`\sin(\sqrt{2}\,x_0 - x_1)` carrier discovery).
       Entry points: ``demo_affine_generators.py``, ``gs_smoke_benchmark.py``
   * - ``examples/gs_charts/``
     - GS-to-charts bridge demos: continuous graph symmetries compiled into
       executable input charts, with blind Sedov-Taylor Trinity yield
       recovery (:math:`R = \xi_0 (E t^2/\rho)^{1/5}`) and SN 1993J dating
       from real VLBI radii.
       Entry points: ``demo_blast_wave.py``, ``demo_sn1993j.py``
   * - ``examples/gs_ablation/``
     - Registry-driven baseline-vs-GS ablation runner for the examples tree
       (records commands, return codes, runtimes, GS reports).
       Entry point: ``runner.py``
   * - ``examples/quadratic_symmetry/``
     - Nonlinear point-symmetry determining equations and invariant
       compilation: :math:`u_{xx} = g/u^3` with the special-conformal
       generator :math:`x^2 \partial_x + x u \partial_u`.
       Entry point: ``conformal_inverse_square.py``
   * - ``examples/special_relativity/``
     - Operational interval-discovery scaffold for Lorentzian kinematics:
       affine boost family, :math:`r = -b/a = \beta`,
       :math:`1/a^2 = 1 - \beta^2`, invariant :math:`u^2 - x^2`.
       Entry point: ``smoke_interval_discovery.py``
   * - ``examples/jacobi_tidal/``
     - Galactic tidal-radius vignette: GS discovery of the anisotropic tidal
       invariant and closed-form Jacobi-radius recovery
       :math:`r_J = \left(\mu/(4\Omega^2 - \kappa^2)\right)^{1/3}`, with a
       standalone note (``jacobi_tidal_note.pdf``; data, logs, and figure
       shipped).
   * - ``examples/kepler_ephemeris_real/``
     - Reduced-Kepler discovery staircase on real heliocentric ephemerides:
       analytic surrogate accelerations on data-discovered cylinder charts,
       a 308-body ensemble with a deterministic 246/31/31 leverage split,
       and a six-panel showcase figure
       (:math:`\dot\theta = h_d/r^2`,
       :math:`\ddot{r} = k_d/r^3 - \mu/r^2`, energy post-pass).
       Entry points: ``smoke_kepler_discovery.py``,
       ``make_direct_paper_figures.py``

Benchmark and Figure Infrastructure
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - ``examples/core_acceptance_suites/``
     - JSON manifests driving ``nestynet_sr/run_core_acceptance_suite.py``:
       explicit mathematical checks protecting the frozen SR/DE core.
       Manifests: ``frozen_core_fast.json``, ``frozen_core_smoke.json``
   * - ``examples/FSS_figure/``
     - Real-data figure explaining factorized-symbolic-search steering in SR
       and DE discovery; every panel is read from archived search reports or
       logs.
       Entry point: ``make_paper_figures.py``

Adjoint and Gradient Tests
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Description
   * - ``tests/test_ast_composite_jvp_vjp_adjoint_audit.py``
     - JVP/VJP adjoint symmetry (error < 1e-12)
   * - ``tests/test_ast_composite_grad_jvp_vjp_adjoint_audit.py``
     - Gradient-level adjoint symmetry
   * - ``tests/test_compound_detection.py``
     - Compound variable detection (product, ratio, power)

Running Examples
----------------

Example smoke scripts and test scripts run as standalone Python scripts from
the repository root::

   python examples/logistic_growth/smoke_logistic_discovery.py --generate
   python tests/test_maxwell_3d.py

Test scripts are also pytest-compatible::

   pytest tests/test_maxwell_3d.py -v

``examples/README.md`` lists a quick-start command sequence for each example
directory (data generation, discovery run, and plotting where applicable).
