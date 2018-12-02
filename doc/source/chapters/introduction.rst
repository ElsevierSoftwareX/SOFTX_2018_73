.. _intro-label:

Introduction
============


.. _objective-label:

Objective
---------

FEniCS Mechanics is a python package developed to facilitate the
formulation of computational mechanics simulations and make these
simulations more accessible to users with limited programming or
mechanics knowledge. This is done by using the tools from the FEniCS
Project (`<www.fenicsproject.org>`_). The FEniCS Project libraries
provide tools to formulate and solve variational problems. FEniCS
Mechanics is built on top of these libraries specifically for
computational mechanics problems.


.. _current-state-label:

Current State
-------------

At this point, FEniCS Mechanics can handle problems in fluid mechanics
and (hyper)elasticity. Fluid Mechanics problems are formulated in
Eulerian coordinates, while problems in elasticity are formulated in
Lagrangian coordinates. Further development is required for the
implementation of Arbitrary Lagrangian-Eulerian (ALE)
formulations. Single domain problems are supported, but interaction
problems are not. While the user may provide their own constitutive
equations, the following definitions are provided by FEniCS Mechanics:

- Solids:

  - Linear Isotropic Material
  - Neo-Hookean Material
  - Fung-type Material
  - Guccione et al. (1995)

- Fluids:

  - Incompressible Newtonian
  - Incompressible Stokes

See section :ref:`user-consteqn-label` for instructions on how to
provide a user-defined constitutive equation.

Also note that this package can handle both steady state and
time-dependent simulations. Check :ref:`finite-difference-label`.


.. _installation-label:

Installation
------------

FEniCS Mechanics requires that version 2016.1.0 or newer of FEniCS be
installed. Installation instructions can be found at
`<http://fenicsproject.org/download>`_.

If you are using FEniCS 2016.1.0 or 2016.2.0 in Python 2, you may also
install the FEniCS Application CBC-Block found at
`<https://bitbucket.org/fenics-apps/cbc.block>`_. Note that to use
CBC-Block, PETSc must be installed with the Trilinos ML package.

Once dependencies have been installed, the user simply has to inform
python of the path where FEniCS Mechanics is stored. This could be
done in several ways. Here are three possible solutions:

#. Add the directory to the :code:`PYTHONPATH` environment variable::

     export PYTHONPATH=<path/to/fenicsmechanics>:$PYTHONPATH

#. Append the directory to the path in your python script

   .. code-block:: python

      import sys
      sys.path.append("path/to/fenicsmechanics")

#. Add a file to the python :code:`dist-packages` directory with the
   :code:`pth` extension, e.g.
   :code:`python3.5/dist-packages/additional_packages.pth` with a
   list of additional directories you want python to check for
   importable libraries. The content of this file will simply be::

     path/to/fenicsmechanics

   with the specific path on your machine.
