Nonlinear DE Point Symmetries
=============================

The generalized-symmetry (GS) layer described in :doc:`generalized_symmetries`
is affine in its chosen chart.  For scalar first- and second-order ODEs,
enabling GS additionally activates a fail-closed *affine-first,
bounded-quadratic-second* policy for point generators

.. math::

   X = \xi(x, u)\,\partial_x + \eta(x, u)\,\partial_u,
   \qquad \deg(\xi), \deg(\eta) \leq 2.

For a fixed polynomial dictionary the determining equation is linear in the
generator coefficients.  The lane is implemented in
``nestynet_sr/sr_gs/nonlinear_de_symmetry.py``, with invariant compilation in
``nestynet_sr/sr_gs/de_invariant_compiler.py`` and matched opportunity
accounting in ``nestynet_sr/sr_gs/nonlinear_opportunity.py``.

Determining Solve and Certification
-----------------------------------

``recover_polynomial_de_symmetries()`` first recovers the on-shell generator
nullspace :math:`N`, then certifies functional relative invariance with the
reduced off-shell solve

.. math::

   \bigl[\operatorname{pr}F\,N,\; -F\,\chi\bigr]\,[a, q]^T = 0,

where :math:`\chi` is a degree-bounded jet-monomial dictionary for the
multiplier :math:`\Lambda(x, u, u_x[, u_{xx}])`.  Degree two is the default
multiplier bound because projective free-particle generators can require
products such as ``x*u_x``.

The report treats the recovered subspace as the scientific object.  It records
its projector and spectrum, held-out residual, bootstrap principal angles,
sparse rotated representatives, representative multipliers, and Lie-bracket
closure in evaluated function space.  Brackets of quadratic fields are allowed
to pass through cubic intermediate expressions.

When bootstraps are requested, their maximum principal angle is a hard
promotion gate rather than advisory metadata.  Individual generators that pass
their determining certificates remain auditable when their degree-truncated
span is nonclosed, but that span is labeled
``recovered_generators_nonclosed`` and is not promoted as a full algebra.

Off-shell certification independently samples ``(x, u[, u_x])`` inside the
observed coordinate ranges rather than reusing paired trajectory points.  The
nonlinear lane refuses promotion when a required coordinate has negligible
span (for example, one equilibrium trajectory); multiple trajectories or
explicit off-trajectory coverage are then required.  This prevents polynomial
fields that merely vanish on a thin observed orbit from being reported as
equation symmetries.

Invariants and Symbolic-Search Hand-Off
---------------------------------------

``compile_point_invariants()`` searches a finite typed AST vocabulary for
low-complexity carriers satisfying :math:`X_a I = 0`, with held-out action,
domain, variance, gradient, and functional-independence gates.  For a
one-generator algebra, ``compile_orbit_coordinate()`` similarly solves
:math:`X s = 1`.

``SymbolicInvariantObjective`` exposes the same certificates as a callable
loss for the external factorized symbolic search.  Certified ASTs can be
handed to:

* ``nonlinear_invariant_de_term_rows()`` (in ``sr_gs/de_bridge.py``) for a
  second DE-library pass;
* ``nonlinear_invariant_carrier_seeds()`` (in
  ``sr_search/factorized_search/gs_carrier_seed.py``) for factorized-search
  carrier seeds.

Automatic routing compiles singleton generator subalgebras as well as the full
algebra.  This matters when the full algebra has only constant common
invariants but a projective or scaling generator has a simple useful carrier.
The full-algebra compilation is disabled when bracket closure fails; stable
singleton generators may still contribute independently certified carriers.
Only certified non-affine carriers can launch the default bounded
factorized-search challenger, and that challenger replaces the baseline only
after held-out improvement.  This split prevents a flexible generator
dictionary from receiving credit for merely tracing solution level sets: a
generator is useful only when it is stable, certified off shell, and yields
simple noncollapsed carriers.

Command-Line Interface
----------------------

The parent switch remains::

   --gs-enable

With GS active, automatic nonlinear escalation and one bounded carrier-seeded
factorized-search (FSS) attempt are on by default.  They can be disabled
independently::

   --gs-de-no-auto-nonlinear
   --gs-de-no-auto-fss

The automatic challenger is capped independently of the broad manual FSS
lane: one attempt, 1500 iterations, 1024 fit/probe points, depth four, and an
eight-candidate shortlist by default (``--gs-de-auto-fss-max-attempts``,
``--gs-de-auto-fss-n-iter``, ``--gs-de-auto-fss-n-fit``,
``--gs-de-auto-fss-n-probe``, ``--gs-de-auto-fss-max-depth``,
``--gs-de-auto-fss-return-topk``).  Explicitly smaller user budgets remain
smaller.

Bounds and certificates are controlled by::

   --gs-de-determining-max-degree 2
   --gs-de-determining-multiplier-degree 2
   --gs-de-determining-bootstraps 8
   --gs-de-nonlinear-invariants
   --gs-de-nonlinear-invariant-max-degree 3

Further controls include ``--gs-de-nonlinear-invariant-max-candidates``,
``--gs-de-nonlinear-invariant-tol``, ``--gs-de-no-orbit-coordinate``,
``--gs-de-no-sparse-rotation``, and ``--gs-de-no-bracket-certificate``.

``--gs-de-determining-equations`` and ``--gs-de-nonlinear-invariants`` remain
available as explicit force-on controls.  Cubic point generators,
contact/nonlocal symmetries, and coupled/PDE prolongations are not part of
this bounded lane.

Candidate-equation anchor elimination is intentionally restricted to residuals
that are affine and monic in the highest jet.  The certificate probes the
anchor derivative away from the elimination point and fails closed for terms
such as ``DU()**2``, rather than constructing the wrong on-shell manifold.

Matched Opportunity Accounting
------------------------------

``nonlinear_opportunity.py`` implements the A/B/C comparison used to decide
whether nonlinear symmetry labels add downstream value:

* **A**: affine-in-adapted-charts GS;
* **B**: the neutral baseline with the same extra carrier vocabulary;
* **C**: quadratic GS with certified generated carriers.

Arm C is credited by ``evaluate_matched_opportunity()`` only when B and C use
exactly the same extra vocabulary, its absolute
determining/stability/certificate gates pass, and it improves invariant
recovery, reduction, or held-out equation/rollout metrics over B.  The
built-in registry includes free-particle/projective, Riccati/Mobius, and
generic negative-control scalar ODE cases; wave and Schrodinger examples are
registered as deferred coupled/PDE cases rather than silently approximated.

Related Symmetry-Reduction Capabilities
---------------------------------------

The bounded quadratic lane sits alongside several other symmetry-driven
reduction tools in ``nestynet_sr/sr_gs/``.

Noether Reduction and Momentum Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``noether_reduction.py`` is the symplectic (Hamiltonian) analogue of the
scalar-ODE reduction cascade.  It scans a small basis of candidate
linear/affine phase-space generators (rotations, translations, dilation) via
``canonical_generators()``; each generator has a momentum map -- the conserved
Noether charge -- computed by ``momentum_map()`` from the symplectic pairing.
``discover_noether_symmetries()`` keeps the generators whose charge is
conserved along the trajectory data, and ``central_force_reduction()`` reduces
by a discovered rotational symmetry: the conserved angular momentum fixes the
orbit plane and eliminating the cyclic angle gives the effective radial
dynamics, so the centrifugal coefficient equals :math:`\ell^2` by
construction rather than by fit.  Nothing about "angle" is presupposed: the
rotation is discovered from Cartesian phase-space data as the symmetry whose
charge the data conserve.

Solvable-Algebra Cascade to Quadrature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A second-order ODE admitting a two-dimensional solvable point-symmetry
algebra integrates by two successive reductions.
``solvable_cascade_reduction()`` in ``de_reduction.py`` discovers a first
generator on the trajectories, reduces the order-2 equation to a first-order
equation :math:`dv/dr = H(r, v)`, then discovers a second symmetry on the
*reduced* ensemble via ``reduced_equation_symmetry()``.  When the reduced
Riccati is autonomous (translation symmetry) the equation belongs to the
constant-coefficient linear family, its equilibria are the characteristic
roots, and ``recognize_constant_coeff_linear_solution()`` returns the
closed-form general solution from the discriminant.

Cauchy-Euler Scale-Invariant Lane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the reduced Riccati is instead scale-invariant under
:math:`(r, v) \to (\lambda r, v/\lambda)` the second symmetry is the scaling
:math:`x\,\partial_x` (equidimensionality) and the original equation belongs
to the Cauchy-Euler family :math:`x^2 u'' + a x u' + b u = 0`.
``recognize_equidimensional_solution()`` reads the indicial equation
:math:`m^2 + (a-1)m + b = 0` off the scale-invariant equilibria
:math:`v = m/r` and returns the power-law general solution, mirroring the
characteristic-root case with :math:`x^m` in place of :math:`e^{mx}` and
:math:`\ln x` in place of :math:`x`.

Certified Casimir Discovery and Algebra Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``algebra_certificates.py`` decides what a recovered affine symmetry subspace
is allowed to do next: ``certify_affine_algebra()`` performs Lie-bracket
closure checks (``BracketCertificate``) and gates whether the subspace remains
audit-only or becomes eligible for quotient/reduction construction;
descriptive generator labels (``classify_affine_generator()``) are reported
for humans but do not control acceptance.

``algebra_casimirs.py`` extracts certified structure constants from recovered
algebras -- ``extract_phase_structure_constants()`` for phase-space
generators, ``extract_affine_structure_constants()`` for affine graph
symmetries -- with an explicit Jacobi-identity residual
(``StructureConstantsCertificate``).  ``discover_algebra_casimirs()`` then
searches the associated Lie-Poisson tensor for Casimir invariants (reusing
the general Poisson machinery of :doc:`poisson_geometry`), and
``certify_charge_brackets()`` checks the anti-homomorphic charge-bracket
relations :math:`\{J_a, J_b\} = -c_{abc} J_c + \kappa_{ab}` on data.  The
overall sign convention is explicit in every charge-bracket report; Casimir
discovery itself is insensitive to it.
