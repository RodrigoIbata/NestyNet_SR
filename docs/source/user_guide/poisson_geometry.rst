General Poisson Geometry
========================

The Poisson discovery layer searches for a skew tensor :math:`\Pi(z)` and a
scalar Hamiltonian :math:`H(z)` satisfying

.. math::

   \dot z=f(z)=\Pi(z)\nabla H(z).

It does not assume an even state dimension or a predefined ``(q,p)`` split.
The canonical Hamiltonian search remains available as a fast path when the
standard symplectic matrix is already known.

Discovery funnel
----------------

Candidate generation starts from the linear determining equation

.. math::

   \mathcal L_f\Pi=0.

Only its skew upper triangle is represented.  The determining nullspace is
reported as a subspace projector with held-out residuals, spectral diagnostics,
and bootstrap principal angles.  Representatives are subsequently required to
pass the nonlinear Jacobi identity and a fixed-tensor Hamiltonian reconstruction
fit.  Casimirs are found from the second linear nullspace problem

.. math::

   \Pi\nabla C=0.

After sparse coefficient thresholding, each Casimir support is refit and
revalidated. Only candidates passing scaled Poisson-flow and, when supplied,
vector-field drift tolerances appear in ``candidates``; failed sparse supports
remain available in ``rejected_candidates`` for audit.

The public ``PoissonSearchConfig`` lanes use one total-degree polynomial basis:

* degree zero: constant Poisson tensors;
* degree one: linear and affine Lie--Poisson tensors;
* degree two: quadratic symbolic tensors.

Polynomial Jacobi certificates assemble every Jacobiator coefficient, avoiding
state-space collocation blind spots.  Learned floating coefficients still carry
numerical uncertainty; exact arithmetic is only claimed after coefficient
snapping and symbolic revalidation.

Shared geometry
---------------

``discover_poisson_structure_multi`` stacks determining matrices from multiple
vector fields.  This identifies a shared tensor while allowing each experiment
to have a fully shared, shared-support, or independent Hamiltonian head.  The
state coordinates and units must be aligned across datasets.

Three-dimensional rank-two lane
-------------------------------

In three dimensions a skew tensor is represented by a vector ``J`` with
``Pi v = J cross v``.  ``reconstruct_nambu_3d`` accepts two sampled scalar
integrals and fits

.. math::

   f=\mu\,\nabla C\times\nabla H,
   \qquad J=\mu\nabla C.

The report identifies singular points where the two gradients lose
independence, separates pointwise multiplier pseudo-targets from differentiable
feature fits, and records the regular-patch reconstruction residual.
The rank-two lane requires nonzero rank on a generic fraction of the regular
patch. A zero vector field is therefore not accepted as a discovered rank-two
structure. Pointwise multipliers remain symbolic-regression pseudo-targets; a
differentiable multiplier representation, explicitly declared through the
feature contract, is required before
``jacobi_by_construction`` and overall acceptance can be true.

Generalized Noether classification
-----------------------------------

``classify_noether_symmetry`` keeps three logically distinct gates:

.. math::

   \mathcal L_Y\Pi=0,\qquad Y(H)=0,\qquad Y=\Pi\nabla G.

Consequently a result distinguishes a Poisson symmetry from a Hamiltonian
symmetry and never treats a sampled local charge as proof of a global momentum
map.  ``canonical_affine_momentum_map`` is the strict canonical fast path; it
rejects affine generators that do not preserve the canonical tensor.

Darboux charts
--------------

``pullback_poisson_tensor`` constructs a tensor from a supplied differentiable
chart and a canonical, possibly degenerate, latent tensor.  Affine, callable,
triangular, and pure-AST triangular charts are supported.  Certification checks
local invertibility, pushforward closure, constant rank, and sampled Jacobi
residuals.  ``rank_darboux_candidates`` ranks a finite, interpretable chart
slate; the implementation deliberately does not train an unconstrained
invertible neural map.
The resulting ``chart_geometry_accepted`` flag certifies local chart geometry,
not agreement with a recovered vector field. The legacy ``accepted`` property
is retained as a compatibility alias with that narrower meaning.

Domain and gauge reporting
--------------------------

Poisson models are local to the supported state-space region used for their
certificates.  Rank-changing points are reported separately.  Tensor/Hamiltonian
scale, additions of functions of Casimirs, and rank-two functional rescalings
are genuine gauges, so comparisons should use normalized coefficients or
subspace projectors rather than raw coefficient vectors.

Automatic autonomous-vector routing
-----------------------------------

``auto_discover_poisson_structure`` and its multi-system counterpart guard the
polynomial lane ladder with state-dimension, sample-count, coordinate-span,
conditioning, tensor-dictionary, dataset, and representative budgets. A guard
failure returns a structured ``skipped`` report and never changes the baseline
vector law.

Explicit ``search_config`` arguments are rebuilt through a trusted whitelist:
they may select an allowed lane subset or tighten certificates, but cannot
disable nonzero-rank, Hamiltonian, or nullspace-stability requirements; loosen
geometric tolerances; disable first-accepted-lane stopping; or exceed nested
STLSQ, bootstrap, representative, and library caps. These are static compute
caps, not a wall-clock timeout.

Recovered systems have two separately reported nullspace tiers. ``exact`` uses
the strict numerical nullspace. ``noise_calibrated`` is available only when the
system-DE result supplies finite derivative residuals below the configured
ceiling. It may propose stable trailing singular vectors, including the
otherwise gapless one-column two-dimensional constant lane, but promotion still
requires held-out/bootstrap stability, Lie-derivative, Jacobi, rank, and
Hamiltonian reconstruction gates at the calibrated scale. Residuals above the
ceiling abstain rather than loosening the exact tier.

``discover_system_de_from_surrogate`` enables this branch by default for a
recovered first-order autonomous vector system. Explicit time/space or
derivative atoms, collapsed state support, unsupported dimensions, and failed
geometry certificates all fail closed. Set
``SystemDESearchConfig.poisson_auto = False`` to disable it, or supply
``poisson_auto_config`` to change the bounded budgets.
