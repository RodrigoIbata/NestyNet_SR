sr\_de -- Differential Equation Discovery
==========================================

The ``sr_de`` module handles DE discovery, system DE discovery,
vector calculus operators, VarPro refinement, and Hamiltonian discovery.

de\_search -- DE Discovery
---------------------------

.. automodule:: nestynet_sr.sr_de.de_search
   :members:
   :undoc-members:

system\_de\_search -- System and Vector DE Discovery
-----------------------------------------------------

.. automodule:: nestynet_sr.sr_de.system_de_search
   :members: discover_system_de_from_surrogate, SystemDESearchConfig, discover_vector_de_from_surrogate, VectorDESearchConfig, VectorSystemDESearchConfig
   :undoc-members:

vector\_ops -- Vector Calculus Macros
--------------------------------------

.. automodule:: nestynet_sr.sr_de.vector_ops
   :members: curl, div, grad, laplacian, advect, Vec, VField
   :undoc-members:

de\_templates -- Template Families
------------------------------------

.. automodule:: nestynet_sr.sr_de.de_templates
   :members:
   :undoc-members:

varpro\_de -- VarPro Refinement
--------------------------------

.. automodule:: nestynet_sr.sr_de.varpro_de
   :members:
   :undoc-members:

complex\_ops -- Complex-Field DE Discovery
-------------------------------------------

.. automodule:: nestynet_sr.sr_de.complex_ops
   :members:
   :undoc-members:

hamiltonian\_search -- Hamiltonian Discovery
---------------------------------------------

.. automodule:: nestynet_sr.sr_de.hamiltonian_search
   :members:
   :undoc-members:

factorized\_de -- Factorized DE Rescue
---------------------------------------

The public entry point is ``factorized_de``; the implementation is split
across the internal ``_factorized_de_explorer``, ``_factorized_de_frontend``,
``_factorized_de_lanes``, ``_factorized_de_operator``,
``_factorized_de_rescue``, and ``_factorized_de_search`` modules.

.. automodule:: nestynet_sr.sr_de.factorized_de
   :members:
   :undoc-members:

poisson\_search -- General Poisson Geometry
--------------------------------------------

.. automodule:: nestynet_sr.sr_de.poisson_search
   :members:
   :undoc-members:

.. automodule:: nestynet_sr.sr_de.poisson_auto
   :members:
   :undoc-members:

.. automodule:: nestynet_sr.sr_de.poisson_3d
   :members:

.. automodule:: nestynet_sr.sr_de.poisson_noether
   :members:

.. automodule:: nestynet_sr.sr_de.poisson_darboux
   :members:
