***********
Description
***********

FEniCS Mechanics is a python module that simplifies the code that a
user must produce in order to run computational mechanics
simulations. Detailed documentation can be found at
https://shaddenlab.gitlab.io/fenicsmechanics/ The module currently
supports single domain solid and fluid mechanics simulations. More
specifically, the available models are as follows:

* Solids:
  - Isotropic
    - Linear Material
    - Neo-Hookean Material
  - Anisotropic
    - Fung-type Material
    - Guccione et al. (1995)
* Fluids:
  - Incompressible Newtonian
  - Incompressible Stokes

However, the user may provide their own constitutive models,
so long as it is a continuum governed by Cauchy's momentum
equation. This module is built entirely on top of FEniCS.
Thus, the user must have FEniCS installed.
[http://fenicsproject.org]


*******
Authors
*******

- Miguel A. Rodriguez (miguelr@berkeley.edu),
- Christoph M. Augustin (christoph.augustin@berkeley.edu)


*******
License
*******

This software is released under the BSD 3-clause license found in
[./LICENSE.txt].


************
Installation
************

Mandatory dependencies:

- FEniCS, version 2016.1.0 or later. Installation instructions for
  FEniCS can be found at [http://fenicsproject.org/download].

Optional dependencies:

- CBC-Block. The source code can be found at
  [https://bitbucket.org/fenics-apps/cbc.block].
  If this package is not installed, you will be unable to use
  MechanicsBlockSolver. NOTE: To use CBC-Block, PETSc must be
  installed with ML.

Once dependencies have been installed, the user simply has to add the
FEniCS Mechanics directory to the PYTHONPATH enviroment variable in
the terminal via

> export PYTHONPATH=<path to fenicsmechanics>:$PYTHONPATH
