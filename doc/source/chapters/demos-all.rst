.. _examples-label:

Examples
========

Here, we show five examples of how to use FEniCS Mechanics. The first
two are steady-state solid mechanics problems, with the second being
an inverse elastostatics formulation. The third example is a
time-dependent two-dimensional fluid mechanics problem. The fourth is
a steady-state solid mechanics problem that is incrementally
loaded. Thus, it is formulated as a time-dependent problem in FEniCS
Mechanics. The last example shows the user how to define their own
constitutive equation.


.. _steady-solid-label:

Steady-State Solid Mechanics
----------------------------

.. include:: demos/square-doc.txt


.. _inverse-elastostatics-label:

Inverse Elastostatics
---------------------

.. include:: demos/unloading-doc.txt


.. _unsteady-fluids-label:

Time-dependent Fluid Mechanics
------------------------------

.. include:: demos/pipe_flow-doc.txt


.. _unsteady-solids-label:

Time-dependent Anisotropic Material
-----------------------------------

.. include:: demos/ellipsoid-doc.txt

.. _user-consteqn-label:

Custom Constitutive Equation
----------------------------

.. include:: demos/custom-doc.txt
