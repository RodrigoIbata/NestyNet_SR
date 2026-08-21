Symbolic Regression
===================

NestyNet_SR discovers analytical expressions :math:`y = f(x_1, \ldots, x_n)`
from data using a two-stage pipeline built on neural network surrogates.

Stage A: Surrogate and Structure Detection
------------------------------------------

Surrogate Training
~~~~~~~~~~~~~~~~~~

A segmented neural network :math:`\hat{f}(x)` is trained on the data using the
NestyNet Levenberg-Marquardt optimizer.  Key parameters:

* ``--fast``: Reduced epochs/segments for quick testing
* ``--ndata_train`` / ``--ndata_val``: Training/validation row budgets
* ``--batch_size``: LM batch size

(Surrogate depth and budgets are managed internally by the Stage-A
hyperparameters; ``--epochs`` / ``--num_segments`` style flags belong to
``nestynet-de``, not ``nestynet-sr``.)

Separability Detection
~~~~~~~~~~~~~~~~~~~~~~

After training, mixed partial derivatives are analysed to detect structure:

**Additive separability**:

.. math::

   \frac{\partial^2 f}{\partial x_i \partial x_j} \approx 0
   \quad\Longrightarrow\quad
   f(x) = g(x_{\mathcal{A}}) + h(x_{\mathcal{B}})

**Multiplicative separability**:

.. math::

   \frac{\partial^2 \log|f|}{\partial x_i \partial x_j} \approx 0
   \quad\Longrightarrow\quad
   f(x) = g(x_{\mathcal{A}}) \cdot h(x_{\mathcal{B}})

**Compound variables** (monomial):

.. math::

   z = \prod_i x_i^{\alpha_i}

detected via rank-1 structure of the Jacobian in log-space.

**Generalized compounds**: linear (:math:`z = a x_i + b x_j`), radial
(:math:`z = \sqrt{x_i^2 + x_j^2}`), and translation-centered coordinates
(:math:`z = x_i - c`), with complexity-prioritized proposal ordering.

The result is an AST (Abstract Syntax Tree) with neural atoms at the leaves.

AST Representation
~~~~~~~~~~~~~~~~~~

Symbolic expressions are represented as trees with typed nodes:

* **AtomNode**: Neural network atom, variable reference, or constant
* **AddNode**: Addition (two children, same dimension)
* **MulNode**: Multiplication (two children, dimensions add)
* **PowNode**: Exponentiation (base node, rational exponent)
* **SinNode**, **CosNode**, **LogNode**, **ExpNode**: Transcendentals (dimensionless arg)

The AST integrates with PyTorch: every node can be evaluated, differentiated,
and optimized within the NestyNet framework.

Stage B: Analytical Rewriting
-----------------------------

factorized symbolic search Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first pass on each neural atom is factorized symbolic search, a two-phase engine:

1. **Phase 1**: Brute-enumerate all expression trees up to a configurable depth
2. **Phase 2**: UCB-guided mutations (replace, wrap-unary, add/mul random,
   residual-guided, prune)

Each skeleton is fitted with the best of 5 mapping families:

* Polynomial
* Power-law
* Pade (rational)
* Sine
* Exponential

Specialized Rewrite Rules
~~~~~~~~~~~~~~~~~~~~~~~~~

After factorized symbolic search, 40+ specialized rules refine remaining atoms:

* **Polynomial rules**: Detect polynomial structure via Taylor analysis
* **Sinusoidal rules**: Detect periodic structure via FFT features
* **Rational rules**: Detect asymptotic structure via Pade approximation
* **Compound-function macros**: High-payoff motifs (sinc, :math:`\sqrt{1 \pm u}`,
  :math:`(1 \pm u)^{-1}`, hypot) using fast linear screening

Rules are applied iteratively with Occam's razor: simpler rewrites are preferred,
and acceptance requires maintaining or improving fit quality.

Y-Transforms and Outer Peel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When separability detection fails on the raw target :math:`y`, the framework
searches over output transformations:

.. math::

   T(y) = f(x) \quad\text{where}\quad T \in \{\text{id}, x^2, \log, 1/x, \ldots\}

15 candidate transforms are tried with fast prescan.  The outer-peel heuristic
automatically suggests non-identity transforms when the identity fails.

Joint Homogeneity Certificates and the Structural Reserve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some targets are homogeneous only in a *group* of variables.  For example

.. math::

   f(x_0, x_1) = x_0^{-1}\, h(x_1 / x_0)

satisfies :math:`f(\lambda x_0, \lambda x_1) = \lambda^{-1} f(x_0, x_1)`
(joint degree :math:`k = -1`) although neither variable is homogeneous by
itself.  Stage A's gradient analysis proposes such groups from the Euler
ratio :math:`r_S(x) = \sum_{i \in S} x_i \,\partial_i f / f`, and the direct
oracle probe (``probe_oracle_scaling`` / ``probe_oracle_scaling_groups`` in
``sr_search/features.py``) verifies a multi-axis proposal by evaluating
:math:`f(\lambda x_S) / f(x)` at several positive scale factors
(:math:`\lambda \in \{0.7, 0.85, 1.2, 1.5\}` by default) and fitting the
common degree from the finite shared rows.  Verification does **not** require
any singleton axis to pass its own scaling test; certified groups are
returned as ``ScaleSpec`` entries with ``oracle_verified=True``,
``oracle_k``, and ``oracle_rel_std``.

Genuinely power-law output transforms declare an exact
``homogeneity_power`` :math:`p` in the y-transform registry
(``sr_search/y_transforms.py``): identity :math:`p=1`, reciprocal
:math:`p=-1`, sqrt :math:`p=1/2`, square :math:`p=2`.  A verified input
degree :math:`k` transports through such a transform to degree
:math:`p \cdot k` (``derive_joint_homogeneity_certificate`` in
``sr_search/ysearch_ranker.py``).  In the example above, ``square`` maps the
degree :math:`-1` target to a degree :math:`-2` problem, which can expose a
ratio/prefactor decomposition that the identity search misses.

During virtual y-transform selection the ordinary ranking is unchanged.  If
the ordinarily selected set contains no jointly certified transform,
``select_virtual_portfolio`` may append the first-ranked *omitted* certified
transform as a single bounded structural reserve, recorded with
``selection_reason = "joint_homogeneity_reserve"``.  At most one extra
transform is admitted, and the certificate is recomputed on the final
identity Stage-A model rather than trusting evidence from an earlier fit.
The reserve only grants one additional branch of compute: the reserved
transform must still pass the normal Stage-A fit, inverse-branch and
structural checks, Stage-B acceptance, rollback rules, CoE witnesses, and
the terminal statistical audit.

Class-SR and Parameter-SR (Multi-Dataset)
-----------------------------------------

When running on multiple datasets (``--filepaths ...``), Class-SR can perform
post-Stage-B joint fitting with shared vs per-dataset parameters:

* ``--class_sr`` enables Class-SR.
* ``--class_cv_threshold`` controls atom-level CV classification (shared vs local).
* ``--class_sr_optimizer {lbfgs,lm_tie}`` selects the joint optimiser backend.

Class-SR performs two complementary steps:

1. **Direct atom sharing**:
   low-CV leaf tags are treated as class-shared parameters.
2. **Parameter-SR (derived invariants)**:
   a small search over scalar leaf quantities identifies low-scatter derived
   combinations across datasets, such as:

   * ``p*q``
   * ``p/q``
   * ``p*(q^2)/r``

These derived invariants are then injected as soft constraints during the
Class-SR joint fit (LBFGS backend).

Parameter-SR knobs:

* ``--no_class_param_sr``: disable derived-invariant discovery
* ``--class_param_sr_max_invariants``: cap number of retained invariants
* ``--class_param_sr_score_threshold``: invariance score threshold
* ``--class_param_sr_penalty_weight``: soft-constraint weight in joint loss
* ``--class_param_sr_max_scalars``: cap scalar search pool size

Notes:

* ``lm_tie`` uses exact linear equality ties for directly shared leaf parameters.
* Derived-invariant soft constraints currently apply to ``lbfgs``; the
  ``lm_tie`` backend currently reports them but does not enforce them.
* Class-SR outputs now include ``derived_invariants`` in the ``*_classSR.json``
  report and human-readable summary.

Configuration
-------------

Key CLI options for symbolic regression::

   nestynet-sr --filepath data.csv \
       --y_units "[1,0,0]" --x_units "[[1,0,0]]"

Resume from checkpoint::

   nestynet-sr --filepath data.csv --resume_from results/my_data.state.pkl

Multi-dataset Class-SR with derived invariants::

   nestynet-sr --filepaths data/exp1.csv data/exp2.csv data/exp3.csv \
       --class_sr \
       --class_sr_optimizer lbfgs \
       --class_param_sr_score_threshold 0.05 \
       --class_param_sr_penalty_weight 1e-2
