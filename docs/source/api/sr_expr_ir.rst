sr\_expr\_ir -- Expression IR
==============================

The ``sr_expr_ir`` module provides opt-in quotient-DAG expression IR
utilities shared by the SR, factorized-search, DE, and GS paths:
canonicalization, bounded e-graph normalization, signature-based pruning, and
bridges to the core and tuple ASTs.

config -- IR Configuration
---------------------------

.. automodule:: nestynet_sr.sr_expr_ir.config
   :members:
   :undoc-members:

qdag -- Quotient-DAG Canonicalization
--------------------------------------

.. automodule:: nestynet_sr.sr_expr_ir.qdag
   :members:
   :undoc-members:

egraph\_normalizer -- Bounded E-Graph Facade
---------------------------------------------

.. automodule:: nestynet_sr.sr_expr_ir.egraph_normalizer
   :members:
   :undoc-members:

signatures -- Dimensional Signature Helpers
--------------------------------------------

.. automodule:: nestynet_sr.sr_expr_ir.signatures
   :members:
   :undoc-members:

core\_bridge / tuple\_bridge -- AST Bridges
--------------------------------------------

.. automodule:: nestynet_sr.sr_expr_ir.core_bridge
   :members:
   :undoc-members:

.. automodule:: nestynet_sr.sr_expr_ir.tuple_bridge
   :members:
   :undoc-members:

stats -- IR Counters
--------------------

.. automodule:: nestynet_sr.sr_expr_ir.stats
   :members:
   :undoc-members:

reporting -- IR Reporting
--------------------------

.. automodule:: nestynet_sr.sr_expr_ir.reporting
   :members:
   :undoc-members:
