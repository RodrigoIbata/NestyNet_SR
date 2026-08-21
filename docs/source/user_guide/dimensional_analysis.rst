Dimensional Analysis
====================

NestyNet_SR optionally enforces dimensional consistency at two levels:
*local* checks that validate individual operations, and *global*
constraint propagation that detects dead-end configurations before
expensive fitting.

Units Specification
-------------------

Each quantity is represented by a dimension vector
:math:`\mathbf{u} \in \mathbb{Q}^B`, where :math:`B` is the number of
base dimensions (e.g. :math:`B = 3` for a :math:`[L, T, M]` basis):

.. math::

   [y] = [L^{a_1}\, T^{a_2}\, M^{a_3}]
   \;\leftrightarrow\;
   \mathbf{u}_y = (a_1, a_2, a_3)

The user supplies :math:`\mathbf{u}_y` for the target,
:math:`\mathbf{u}_{x_i}` for each input variable, and optionally
:math:`\mathbf{u}_c` for free/fixed constants.

All exponents are exact rationals (Python ``Fraction``) so that fractional
powers like :math:`\sqrt{\cdot}` are handled without floating-point error.

Dimensional analysis is enabled by default when a units spec is provided.
To disable it, pass ``--ignore_units``::

   nestynet-sr --filepath data.csv --ignore_units

Level 1: Local Consistency Checks
----------------------------------

Four local gates validate individual operations during the search:

* **Addition**: operands must have matching dimensions
* **Multiplication**: dimension exponents add:
  :math:`[\text{parent}] = [\text{left}] + [\text{right}]`
* **Exponentiation**: :math:`[\text{parent}] = s \cdot [\text{base}]`
  where :math:`s \in \mathbb{Q}` is the exponent
* **Transcendentals** (:math:`\log`, :math:`\exp`, :math:`\sin`, :math:`\cos`):
  argument must be dimensionless; output is dimensionless

These checks are applied at four decision points:

1. **Full AST validation** after Stage B rewrites (``check_units_ast()``)
2. **Buckingham** :math:`\pi`\ **-count** for compound-variable proposals (``check_compound_buckingham()``)
3. **Single-node feasibility** for additive/multiplicative splits (``check_split_feasibility()``)
4. **Output-dimension inference** for individual atoms (``infer_atom_output_dim()``)

Each gate examines its own local neighbourhood: "Is this compound/split/rewrite
dimensionally legal in isolation?"  But it does not ask: "Given this choice, can
the *rest* of the tree still be satisfied?"

Level 2: Global Constraint Propagation (Buckingham-Sudoku)
-----------------------------------------------------------

Motivation
~~~~~~~~~~

Local checks can approve a configuration that is *globally* infeasible.

Consider a multiplicative decomposition
:math:`\text{AST} = \text{NN}[x_0] \times (\text{NN}[x_1, x_3] \times \text{NN}[x_2, x_4])`
for a problem where :math:`[x_1] = [x_3] = [L]`.  A compound proposal
:math:`z = x_1 / x_3` is dimensionless, and the local Buckingham check passes.

However, once the inner atom sees only a dimensionless input, it can only
produce dimensionless output under span semantics.  The outer multiplication
then cannot reach the required target dimension --- a dead end that Level 1
discovers only after expensive fitting.

Level 2 detects such dead ends *before any fitting* by solving the full
dimensional constraint system globally.

The Sudoku Analogy
~~~~~~~~~~~~~~~~~~

The method is best understood through a Sudoku analogy:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sudoku concept
     - Dimensional analogue
   * - Cell
     - AST node's output dimension
   * - Given (clue)
     - Known dimension (variable, constant, or target)
   * - Pencil marks
     - Feasible dimension subspace (``DimSubspace``)
   * - Row/col/box rule
     - Dimensional constraint (Add = intersect, Mul = sum, etc.)
   * - Naked single
     - Node whose dimension is uniquely determined (rank-0 subspace)
   * - Dead end (empty cell)
     - Infeasible configuration (empty feasible set)

DimSubspace: The Pencil Marks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each AST node is assigned a ``DimSubspace`` --- an affine subspace of
dimension-space representing all dimensions the node *could* take:

.. math::

   \mathcal{S} = \bigl\{\,
     \mathbf{d}_0 + \sum_i c_i\,\mathbf{b}_i
     \;\big|\; c_i \in \mathbb{Q}
   \,\bigr\}

where :math:`\mathbf{d}_0 \in \mathbb{Q}^B` is an offset (particular solution)
and :math:`\{\mathbf{b}_i\}` is a basis for the free directions.

Three special states:

* **Pinned** (basis empty): dimension uniquely determined --- a "naked single"
* **Unconstrained** (no offset): all dimensions possible --- blank cell
* **Empty** (system inconsistent): no dimension feasible --- dead end

The ``DimSubspace`` dataclass::

   @dataclass(frozen=True)
   class DimSubspace:
       offset: Dim              # affine offset (particular solution)
       basis: tuple[Dim, ...] = ()   # free directions

Constraint Rules
~~~~~~~~~~~~~~~~

Each AST node type imposes a linear constraint.  The bottom-up ("what can a
subtree produce?") and top-down ("what must a subtree produce?") rules are:

.. list-table::
   :header-rows: 1
   :widths: 15 35 35

   * - Node
     - Bottom-up (achievable)
     - Top-down (required for child)
   * - Add
     - :math:`\mathcal{S}_L \cap \mathcal{S}_R`
     - child :math:`\leftarrow \mathcal{S}_{\text{parent}}`
   * - Mul
     - :math:`\mathcal{S}_L \oplus \mathcal{S}_R`
     - child :math:`\leftarrow \mathcal{S}_{\text{parent}} \ominus \mathcal{S}_{\text{sibling}}`
   * - Pow(s)
     - :math:`s\,\mathcal{S}_{\text{base}}`
     - base :math:`\leftarrow \tfrac{1}{s}\,\mathcal{S}_{\text{parent}}`
   * - log/exp/sin
     - :math:`\{\mathbf{0}\}`
     - child :math:`\leftarrow \{\mathbf{0}\}`
   * - Leaf
     - from kind
     - (no children)

Here :math:`\oplus` denotes the Minkowski sum
:math:`\mathcal{A} \oplus \mathcal{B} = \{a + b \mid a \in \mathcal{A},\, b \in \mathcal{B}\}`
and :math:`\ominus` is the analogous difference.

Algorithm: Solve via RREF in One Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because every constraint is linear over :math:`\mathbb{Q}`, the entire
"Sudoku game" reduces to a single linear system.

An AST walk assigns each node a symbolic *DimExpr*:

.. math::

   [\text{node}] = \mathbf{c}_0 + \sum_i \alpha_i\,\mathbf{U}_i

where :math:`\mathbf{c}_0` is a known constant part and each
:math:`\mathbf{U}_i` is an unknown dimension vector (one per unconstrained
NN atom).  Equality constraints from AddNodes, the root target, and
transcendental-dimless requirements form a system
:math:`\mathbf{A}\,\mathbf{x} = \mathbf{b}` over :math:`\mathbb{Q}`.

Solving via reduced row echelon form (RREF) yields:

.. math::

   \mathbf{x} = \mathbf{x}_0 + \sum_j t_j\,\mathbf{n}_j,
   \qquad t_j \in \mathbb{Q}

Each node's ``DimSubspace`` is obtained by substituting this solution
into the node's DimExpr and projecting:

1. **Offset**: evaluate the DimExpr at :math:`\mathbf{x}_0`
2. **Basis**: for each null-space vector :math:`\mathbf{n}_j`, evaluate
   the DimExpr's linear part to get a direction; reduce to independent set

If RREF reveals an inconsistency (:math:`0 = c \neq 0`), the entire AST is
infeasible --- dead end detected without any fitting.

The computation is exact (rational arithmetic), takes :math:`O(N B^2)` time,
and is sub-millisecond for the :math:`N \leq 50` trees encountered in practice.

API::

   from nestynet_sr.sr_core.units import compute_node_domains, DimSubspace

   domains = compute_node_domains(root_ast, units_spec)
   # domains: dict[int, DimSubspace] or None if infeasible

Worked Example
~~~~~~~~~~~~~~

Consider the multiplicatively-decomposed AST for a problem with target
dimension :math:`[y]` and atoms :math:`\text{NN}[x_1, x_3]` where
:math:`[x_1] = [x_3] = [L]`:

1. **Before hypothesis**: :math:`\text{NN}[x_1, x_3]` has achievable
   subspace :math:`\text{span}\{[L]\}` (rank 1).  RREF solves the global
   system; the node's required dimension is non-trivially constrained
   but compatible with :math:`[L]^n` for some :math:`n`.

2. **Hypothesis** :math:`z = x_1/x_3`: the compound is dimensionless,
   so :math:`\text{NN}[z]` can only produce :math:`\{\mathbf{0}\}`.

3. **RREF**: the constraint system now requires this node to produce a
   non-dimensionless value, but the achievable set is :math:`\{\mathbf{0}\}`.
   Intersection is empty --- **dead end detected** before any fitting.

Hypothesis Testing
------------------

Level 2 enables cheap "what-if" tests for candidate structural changes.

Compound Proposals
~~~~~~~~~~~~~~~~~~

"What if atom A uses compound :math:`z = x_i / x_j`?"

1. Build a modified AST with the compound applied
2. Solve the constraint system from scratch via RREF
3. Check for empty feasible sets
4. If any node is empty, reject the proposal

::

   from nestynet_sr.sr_core.units import propose_split, compute_node_domains

   # Returns per-node domains or None if infeasible
   domains = compute_node_domains(root, spec)
   result = propose_split(root, spec, atom)

Split Proposals
~~~~~~~~~~~~~~~

"What if we split atom A additively on :math:`\{x_0, x_1\} \mid \{x_2, x_3\}`?"

Same procedure with the split applied::

   from nestynet_sr.sr_core.units import propose_split

   result = propose_split(root, spec, atom, "add", group1=[0,1], group2=[2,3])

Non-Mutating Design
~~~~~~~~~~~~~~~~~~~~

Each test is non-mutating: a fresh candidate AST is constructed, the global
system is solved, and the result is inspected.  The original AST is untouched
if the hypothesis fails.

Level 1 serves as a fast pre-filter (:math:`O(1)` Buckingham :math:`\pi`-count),
and Level 2 serves as a global validator (:math:`O(NB^2)` RREF solve).
Testing 20 hypotheses takes ~1 ms total, negligible compared to the minutes
of fitting saved by skipping infeasible configurations.

Integration with the Search
----------------------------

Level 2 integrates at three points in the search pipeline:

1. **After Stage A**: Build the constraint graph from the current AST.
   If infeasible, skip the configuration entirely.

2. **Compound screening**: After Level 1 pre-filter passes, run Level 2
   hypothesis test.  Reject if globally infeasible.

3. **Split screening**: Same pattern for additive/multiplicative split
   proposals.

Edge Cases
----------

* **Both MulNode children unknown**: Minkowski sum of two full spaces is
  full space (no narrowing), but propagation remains correct.  Narrowing
  occurs once one child gets rewritten.

* **Tag-sharing**: Atoms with the same tag share a single unknown in the
  linear system, so tag-sharing cycles are handled naturally.

* **Dimensionless target**: All constraints become trivially satisfiable
  under ``nn_semantics="unknown"``.

* **Free constants expanding spans**: Under the permissive policy (default),
  NN parameters can carry units.  A strict policy (``UnitsSpec.nn_semantics``, CLI ``--nn_units_semantics`` / ``--units_policy``)
  enables more aggressive pruning.
