Application Program Interface
=============================

The difference classes in FEniCS Mechanics and member classes are
listed here.

Problem Objects
---------------

The problem classes defining the weak form for mechanics problems are
listed here.

Base Mechanics Problem
**********************

All mechanics problem objects are child classes of
:code:`basemechanicsproblem.BaseMechanicsProblem`.

.. automodule:: fenicsmechanics.basemechanicsproblem

   .. autoclass:: BaseMechanicsProblem
      :members:


Mechanics Problem
*****************

.. automodule:: fenicsmechanics.mechanicsproblem

   .. autoclass:: MechanicsProblem
      :members:


Solid Mechanics Problem
***********************

.. automodule:: fenicsmechanics.solidmechanics

   .. autoclass:: SolidMechanicsProblem
      :members:


Fluid Mechanics Problem
***********************

.. automodule:: fenicsmechanics.fluidmechanics

   .. autoclass:: FluidMechanicsProblem
      :members:


Solver Objects
--------------

Mechanics Solver
****************

.. automodule:: fenicsmechanics.mechanicssolver

   .. autoclass:: MechanicsBlockSolver
      :members:


Solid Mechanics Solver
**********************

.. automodule:: fenicsmechanics.solidmechanics

   .. autoclass:: SolidMechanicsSolver
      :members:


Fluid Mechanics Solver
**********************

.. automodule:: fenicsmechanics.fluidmechanics

   .. autoclass:: FluidMechanicsSolver
      :members:


Constitutive Equations
----------------------

The :code:`materials` submodule implements some common constitutive
equations for users to take advantage of.


Solid Materials
***************

.. automodule:: fenicsmechanics.materials.solid_materials
   :members: LinearIsoMaterial, NeoHookeMaterial, FungMaterial, GuccioneMaterial


Fluids
******

.. automodule:: fenicsmechanics.materials.fluids
   :members: NewtonianFluid
