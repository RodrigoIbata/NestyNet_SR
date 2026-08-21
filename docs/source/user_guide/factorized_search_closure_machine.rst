factorized symbolic search: The Typed Closure Machine
=====================================================

factorized symbolic search discovers symbolic expressions by iteratively building a basis of
typed closures, each fitted directly on real data. It does not use genetic
programming, neural network surrogates, or pseudo-targets. The core loop is:

1. Propose a nonlinear hypothesis (a typed closure with slots)
2. Fit coefficients directly against the real target *y*
3. Inspect the residual
4. Propose the next block

This document explains the architecture end-to-end.


Overview
--------

The search has three phases that run in sequence:

- **Brute enumeration**: Exhaustively score all expression trees up to a
  configurable depth. This produces an initial pool of simple candidates and
  establishes the best-known MSE baseline.

- **Closure search**: The typed closure machine. Proposes structured
  nonlinear hypotheses from a small operator algebra, scores each by fitting
  a separable linear head on real *y*, and grows an additive basis of
  accepted blocks. This is where the new architecture lives.

- **Mutation search**: UCB-guided random mutations (replace subtree, wrap
  with unary, add/multiply random term, prune). Refines candidates
  discovered by brute and closure search.

The closure search runs once, between brute and mutation. If it finds an
exact solve (MSE < ``early_stop_mse``), mutation is skipped entirely.


The Operator Algebra
--------------------

The closure machine uses a small typed operator algebra — seven typed families (six enabled by default) that
cover the structural patterns needed for scientific expressions:

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Family
     - Template
     - Head solver
     - Example
   * - **periodic**
     - ``cos(h)`` or ``sin(h)``
     - harmonic linear: ``[a·cos(h), b·sin(h), companions, 1]``
     - ``2·√(x₀x₁)·cos(x₂) + x₀ + x₁``
   * - **exp**
     - ``exp(h)``
     - linear: ``[exp(h), anchor, 1]``
     - ``1.3·exp(x₀x₃) + x₁``
   * - **log**
     - ``log(h)``
     - linear: ``[log(h), anchor, 1]``
     - ``log(x₀x₁) + x₂``
   * - **rational**
     - ``(a₀ + a·u) / (1 + b·v)``
     - fractional linear
     - ``(1 + x₀) / (1 + x₀x₁)``
   * - **power**
     - ``h^p``, ``p ∈ {-2, -1, -½, ½, 1, 2}``
     - discrete power: ``[h^p, anchor, 1]``
     - ``x₀ / √(1 + x₁²)``
   * - **quadratic**
     - ``√(Σ bᵢ²)``
     - quadratic sqrt: ``[√(Σ wᵢbᵢ²), anchor, 1]``
     - ``q · √(E₁² + E₂² + E₃²)``
   * - **affine**
     - ``Σ cᵢtᵢ + c₀``
     - linear
     - ``1.5x₀ - 2x₁ + 0.25``

Each family is defined by an ``OperatorSpec`` (in ``operator_specs.py``)
that encodes:

- A **parent template** with ``__CARRIER__`` and ``__ANCHOR__`` holes
- A **composition mode**: base, companion (additive), or prefactor
  (multiplicative)
- **Slot constraints**: dimensional rules, domain rules, arity caps
- A **carrier dimension resolver** that computes the required carrier
  dimension from the target and anchor dimensions
- A **scaffold ID builder** that generates carrier-specific identity keys


Seed Blocks and Recursive Pool Construction
--------------------------------------------

Before proposing closures, the system builds a pool of candidate
sub-expressions to fill the operator slots. This is done by
``build_recursive_seed_pool()`` in ``seed_blocks.py``.

**Base seeds** come from:

- Raw variables: ``x₀, x₁, ..., xₙ``
- The constant ``1``
- Pool nodes from brute enumeration (products, trig, etc.)

**Recursive builders** extend the pool over multiple rounds:

- **Product builder**: pairwise products ``xᵢ·xⱼ`` (up to arity 3)
- **Monomial builder**: ``xᵢ^p`` for ``p ∈ {-1, -½, ½, 1, 2}``,
  applied to all current pool entries including products from earlier rounds.
  This is how ``√(x₀x₁)`` is constructed.
- **Quadratic builder**: sums of squares ``Σ bᵢ²``, with dimensional
  bucketing. Bases are grouped by dimension; combinations are only generated
  within same-dimension buckets. When the operator provides a required
  carrier dimension, only bases whose ``2·dim = required_dim`` are admitted.
  Arities are processed in descending order (3, 2, 1) so that multi-term
  norms like ``x₁² + x₂² + x₃²`` are tried before single-term ``x₁²``.
- **Affine builder**: linear combinations of pool entries

Each seed block carries:

- The AST node
- Dimensional signature (or ``None`` if dimensions aren't enforced)
- Domain tags (``positive``, ``nonnegative``, etc.)
- Builder metadata (depth, nonlinear depth, product arity)


Operator Enumeration
--------------------

``enumerate_operator_applications()`` in ``scaffold_enum.py`` generates
concrete proposals by binding seed blocks to operator slots:

1. Build **anchor blocks** from the pool (prioritised by simplicity)
2. Build **carrier blocks** from the recursive seed pool
3. For each operator spec in each family:

   a. Find compatible (anchor, carrier) pairs using dimensional and domain
      constraints
   b. Construct a ``BoundClosure`` with the bindings
   c. Emit an ``OperatorApplication`` with a carrier-specific scaffold ID

The enumeration respects per-family **budget caps** from the steering
allocator. Each family gets a fair-share floor of the total budget, with
remainder distributed by residual-guided priority scores.


Two-Lane Architecture
---------------------

The closure search runs in two lanes to prevent basis-derived state from
crowding out canonical typed proposals:

**Core lane** (runs first):

- Uses a clean, deterministic seed pool from ``build_pool(nvars)``
- No basis state, no mutation-derived pool nodes
- Produces the canonical typed proposals (e.g., the exact norm
  ``q·√(E₁²+E₂²+E₃²)`` for a quadratic problem)

**Augmented lane** (runs second, optional):

- Uses the full boost pool including mutation-discovered expressions
- Adds basis-derived seed blocks via ``extend_seed_blocks_with_basis()``
- Can propose closures over learned features, not just raw variables

The two lanes are merged **after** direct scoring, not before enumeration.
This ensures that typed proposals from the core lane are never crowded out
by basis augmentation.


Direct Scoring
--------------

Each proposal is scored by ``solve_direct_operator_preview_rows()`` in
``direct.py``. The process:

1. **Resolve a planner** from the ``DIRECT_OPERATOR_PLANNERS`` registry
   based on the operator spec's family and kind
2. The planner **builds a design matrix** from the closure's terms:

   - For ``cos(h) + anchor + 1``, the columns are
     ``[cos(h_fit), anchor_fit, ones]``
   - For a rational ``(a₀+a·u)/(1+b·v)``, the columns encode the
     fractional form

3. **Fit the linear head** in closed form via ``torch.linalg.lstsq()``.
   No iterative optimisation. This gives coefficients and MSE in one shot.
4. **Materialize the expression**: the fitted closure is rendered as an
   AST. Crucially, the materialiser emits **structural terms only** — no
   embedded ``const`` coefficients. This preserves dimensional consistency:
   ``cos(x₂) + x₀`` instead of ``2.0·cos(x₂) + 1.0·x₀ + 0.5``.
   The coefficients live in the head mapping metadata.
5. **Dimensional check**: the materialised expression must pass
   ``node_dims()`` consistency. If the dimensions don't match the target,
   the candidate is rejected.

The head solver registry supports:

- ``linear``: standard OLS
- ``harmonic_linear``: ``[a·cos, b·sin, companions, 1]``
- ``fractional_linear``: ``(a₀ + a⊤u) / (1 + b⊤v)``
- ``quadratic_sqrt``: ``√(Σ wᵢ·bᵢ²)``
- ``discrete_power``: ``(a₀ + a₁·h)^p`` for discrete ``p``


Basis State and Iterative Construction
--------------------------------------

Accepted closures become **feature blocks** in a ``BasisState``:

.. code-block:: python

   BasisState(
       blocks=(FeatureBlock(...), FeatureBlock(...), ...),
       fit_loss=...,        # joint fit error
       probe_loss=...,      # generalisation error
       compiled_expr=...,   # final additive expression
       residual_fit=...,    # y - ŷ on fit split
       residual_probe=...,  # y - ŷ on probe split
   )

Each ``FeatureBlock`` records:

- The closure's family and atoms (AST nodes)
- Latent bundle (intermediate features) and head bundle (head terms)
- Active variables and metadata

The **iterative basis loop** in ``run_closure_search_pass()`` works as:

1. Score all preview candidates from the runner
2. Fast-track any candidate with MSE < ``early_stop_mse``
3. For each remaining candidate (ranked by preview MSE):

   a. Create a ``FeatureBlock`` from the closure
   b. Append to the current ``BasisState``
   c. **Global refit**: ``fit_basis_state_head()`` re-fits the linear
      coefficients of all blocks jointly via least squares
   d. **Backward pruning**: drop blocks whose coefficients are near zero
   e. **Subset pruning**: exhaustive search for the minimal block set

4. Admit the refined state to a **basis beam** (top-k by probe loss)
5. If the beam improved, rebuild the residual-guided context and restart
   enumeration from the updated state

This means the basis can grow from zero to multiple blocks in one pass,
with each addition globally refit and pruned.


Dimensional Analysis
--------------------

Dimensional consistency is enforced at every level:

- **Seed blocks** carry dimensional signatures
- **Quadratic builder** uses dimensional bucketing: only combines bases
  with matching dimensions
- **Operator specs** define carrier dimension resolvers:
  ``carrier_dim = f(target_dim, anchor_dim, dim0)``
- **Direct scoring** checks that the materialised expression has
  ``node_dims(expr) == y_dims``
- **Materialiser** emits structural terms only (no ``const`` coefficients)
  so that ``add`` of differently-dimensioned terms is never constructed

This is a hard filter, not a soft penalty. Dimensionally inconsistent
proposals are rejected, not down-weighted. For AI Feynman problems with
physical units, this eliminates the vast majority of the search space.


Configuration
-------------

Key hyperparameters in ``FactorizedSearchConfig``:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Parameter
     - Default
     - Description
   * - ``closure_search_enable``
     - ``False``
     - Enable the closure machine
   * - ``closure_search_families``
     - periodic, exp, log, rational, power, quadratic (``affine`` opt-in)
     - Which operator families to use
   * - ``closure_search_max_proposals``
     - ``16``
     - Total scaffold budget per pass
   * - ``closure_search_anchors_per_family``
     - ``4``
     - Carrier/anchor candidates per family
   * - ``closure_search_preview_topk``
     - ``4``
     - Candidates to preview-score
   * - ``closure_search_exact_topk``
     - ``2``
     - Candidates to exact-score and admit


Example: Feynman 037
---------------------

Target: ``x₀ + x₁ + 2·√(x₀x₁)·cos(x₂)``

The closure machine solves this in one shot:

1. **Seed pool** builds ``√(x₀x₁)`` via the monomial builder applied to
   the product seed ``x₀·x₁``
2. **Periodic operator** proposes ``cos_mul`` with carrier ``x₂`` and
   envelope ``√(x₀x₁)``
3. **Harmonic linear head** fits ``[a·cos(x₂)·√(x₀x₁), b·sin(x₂)·√(x₀x₁), c·x₀, d·x₁, e]`` against *y*
4. Coefficients snap to ``a=2, b=0, c=1, d=1, e=0``
5. Materialised expression: ``((2·cos(x₂))·√(x₀x₁)) + x₀ + x₁``
6. MSE: 5.7×10⁻³⁰ (machine precision). Early-stop triggered.


Example: Feynman 090
---------------------

Target: ``q · √(E₁² + E₂² + E₃²)``

1. **Quadratic builder** with target-aware dimensional bucketing:
   ``required_carrier_dim = target_dim - anchor_dim(q)``. Only bases with
   matching dimension (``E₁, E₂, E₃``) are admitted. Descending arity
   produces the 3-term norm ``sqr(E₁) + sqr(E₂) + sqr(E₃)`` first.
2. **Quadratic sqrt_mul operator** binds carrier to the norm, anchor to
   ``q``
3. **Quadratic sqrt head** fits ``q · √(w₁E₁² + w₂E₂² + w₃E₃²)``
4. Coefficients: ``w₁ = w₂ = w₃ = 1``
5. Materialised expression: ``√(sqr(E₁) + sqr(E₂) + sqr(E₃)) · q``
6. MSE: 1.9×10⁻²⁹. Early-stop triggered.
