.. _math-formulation-label:

Mathematical Formulation
========================

A brief review of continuum mechanics, the finite element method for
spatial discretization, and finite difference methods for time
derivative discretization is presented here. For further details, the
reader is referred to Chadwick :cite:`chadwick_1999` and Hughes
:cite:`hughes_2007`.


.. _continuum-mechanics-label:

Continuum Mechanics
-------------------

The body of interest is assumed to occupy a subregion of
three-dimensional Euclidean space at some initial time, :math:`t_0`,
and denote with region by :math:`\mathcal{B}_0`. We refer to
:math:`\mathcal{B}_0` as the reference configuration. The region
occupied by the same body at a later time, :math:`t`, will be denoted
by :math:`\mathcal{B}`, and referred to as the current
configuration. Solving mechanics problems involves solving for the map
(or its derivatives) that transforms the body from
:math:`\mathcal{B}_0` to :math:`\mathcal{B}`. We will denote this map
with :math:`\boldsymbol\varphi:\mathcal{B}_0\times\mathbb{R}
\rightarrow \mathcal{B}`, where :math:`t \in \mathbb{R}`.

The formulation of continuum mechanics that we will use is concerned
with the deformation of the body at a local level, which is quantified
by the deformation gradient

.. math::

   \mathbf{F}(\mathbf{X}, t) = \frac{\partial\boldsymbol\varphi}
         {\partial\mathbf{X}},

where :math:`\mathbf{X} \in \mathcal{B}_0` is the position vector of a
material point at time :math:`t_0`. This tensor is used to define
other tensors quantifying different strain measures -- all of which
can be used to study the kinematics of deformation.

Once the kinematics of the deformation undergone by the body of
interest has been mathematically formulated, they can be used to study
the relationship between deformation and forces that either cause or
result from it. These relations are referred to as dynamics. This
leads to the concept of the Cauchy stress tensor, which we will denote
by :math:`\mathbf{T}`. The Cauchy stress tensor maps the unit normal
vector, :math:`\mathbf{n}`, on any surface within a continuum to the
traction vector, :math:`\mathbf{t}`, which is the force per unit area
at the point where :math:`\mathbf{T}` and :math:`\mathbf{n}` are
evaluated. By Newton's second law, and the Reynolds Transport Theorem,
we get

.. math::

   \int_{\mathcal{B}} \rho\frac{d\mathbf{v}}{dt}\;dv =
   \int_{\partial\mathcal{B}}\mathbf{t}\;da + \int_{\mathcal{B}}
   \rho\mathbf{b}\;dv,

where :math:`\mathbf{v} = \frac{d}{dt}\mathbf{u}` is the velocity
field of the body, :math:`\mathbf{u}` is the displacement,
:math:`\mathbf{b}` is the force applied per unit mass, and
:math:`\partial \mathcal{B}` is the boundary of
:math:`\mathcal{B}`. Making use of the Cauchy stress tensor, the
divergence theorem, and the localization theorem, we get

.. math::

   \rho\dot{\mathbf{v}} = \text{div}\;\mathbf{T} + \rho\mathbf{b},

where we have replaced the time derivative by the dot, and div is the
divergence operator with respect to the spatial coodinates. FEniCS
Mechanics uses this equation to formulate fluid mechanics problems.
The equivalent equation in the reference coordinate system is

.. math::

   \rho_0\dot{\mathbf{v}} = \text{Div}\;\mathbf{P} +
   \rho_0\mathbf{b},

where :math:`\rho_0 = \rho J` is the referential mass density,
:math:`\mathbf{P} = J\mathbf{TF}^{-T}` is the first Piola-Kirchhoff
stress tensor, :math:`J = \det\mathbf{F}` is the Jacobian, and Div is
the divergence operator with respect to the referential coordinates,
:math:`\mathbf{X}`. This equation is used to formulate solid mechanics
problems.

Note that we have not limited the formulation to a specific type of
material, e.g. rubber or water, and thus these equations are
applicable to all materials that satisfy the continuity assumptions of
the classical continuum mechanics formulation. Specific relations
between strain tensors and the Cauchy tensor are known as constitutive
equations and depend on the material being modelled. For further
details on constitutive equations, the reader is referred to Chadwick
:cite:`chadwick_1999`.


.. _local-constraints-label:

Incompressibility Constraint
****************************

Many materials are modeled as incompressible. In order to satisfy
incompressibility, we must formulate it as a constraint that the
displacement and/or velocity must satisfy. This results in an
additional equation, making the system over-determined. Thus, we need
an additional variable, a Lagrange multiplier, to solve the
system. This Lagrange multiplier ends up being the pressure within the
material, be it a solid or fluid.

If a material is modeled as incompressible, the stress tensor will now
depend on the displacement field for solids, velocity field for
fluids, and the pressure for both. I.e.,

.. math::

   \mathbf{P} = \mathbf{G}(\mathbf{u}, p)

for solids, and

.. math::

   \mathbf{T} = \mathbf{H}(\mathbf{v}, p)

for fluids. The constraint equation will also depend on the relevant
vector field, and the scalar field :math:`p`. Thus,

.. math::

   G(\mathbf{u}, p) = 0

or

.. math::

   H(\mathbf{v}, p) = 0.

The constraint equation for solid materials is of the form

.. math::

   G(\mathbf{u}, p) = \phi(\mathbf{u}) - \frac{1}{\kappa}p,

where :math:`\kappa` is the bulk modulus. For linear materials, we
take

.. math::

   \phi(\mathbf{u}) = \text{div}\;\mathbf{u}.

On the other hand,

.. math::

   \phi(\mathbf{u}) = \frac{1}{J} \ln(J)

is the default expression for nonlinear materials.

The incompressibility constraint for fluid mechanics is

.. math::

   H(\mathbf{v}, p) = \text{div}\;\mathbf{v}.


.. _fem-label:

Finite Element Method
---------------------

The finite element method (FEM) requires a variational form. Thus, we
must convert the governing equations from
:ref:`continuum-mechanics-label` to a variational form. This involves
taking the dot product of the equation of interest with an arbitrary
vector field, :math:`\boldsymbol\xi`, using the product rule, and
divergence theorem to obtain

.. math::

   \int_{\mathcal{B}}\boldsymbol\xi\cdot\rho\dot{\mathbf{v}}\;dv
     + \int_{\mathcal{B}}\frac{\partial\boldsymbol\xi}
       {\partial\mathbf{x}}\cdot\mathbf{T}\;dv
     = \int_{\mathcal{B}}\boldsymbol\xi\cdot\rho\mathbf{b}\;dv
     + \int_{\Gamma_q}\boldsymbol\xi\cdot\bar{\mathbf{t}}\;da,

where :math:`\bar{\mathbf{t}}` is a specified traction on
:math:`\Gamma_q \in \partial\mathcal{B}`. Applying the chain rule to
the material time derivative of the velocity yields

.. math::

   \int_{\mathcal{B}}\boldsymbol\xi\cdot\rho
       \frac{\partial\mathbf{v}}{\partial t}\;dv
     + \int_{\mathcal{B}}\boldsymbol\xi\cdot
       \frac{\partial\mathbf{v}}{\partial\mathbf{x}}\mathbf{v}\;dv
     + \int_{\mathcal{B}}\frac{\partial\boldsymbol\xi}
       {\partial\mathbf{x}}\cdot\mathbf{T}\;dv
     = \int_{\mathcal{B}}\boldsymbol\xi\cdot\rho\mathbf{b}\;dv
     + \int_{\Gamma_q}\boldsymbol\xi\cdot\bar{\mathbf{t}}\;da.

Note that this is the most general variational form of the governing
equation for the balance of momentum, and can thus be used to
provide a general formulation for computational mechanics
problems. FEniCS Mechanics does just this, and changes the physical
meaning of the variables in the above weak form when necessary. E.g.,
the stress tensor :math:`\mathbf{T}` is replaced by the first
Piola-Kirchhoff stress tensor :math:`\mathbf{P}` when formulating a
solid mechanics problem.

Suppose that the true solution to the variational forms belongs to a
function space :math:`\mathcal{F}`. The FEM uses a subspace of
:math:`\mathcal{F}` with finite size to approximate the solution to
the boundary value problem. We denote this finite function space by
:math:`\mathcal{F}^h`, which is spanned by polynomials of order
:math:`n \in \mathbb{N}`, with :math:`n \leq 2` in most cases. Thus,
the approximation of a function :math:`\mathbf{u}` would be a linear
combination of a set of polynomials, i.e.

.. math::

   \mathbf{u} = \sum_{i = 1}^{N_n} \hat{\mathbf{u}}_i
         \phi_i(\mathbf{x}),

where :math:`\hat{\mathbf{u}}_i` are the coefficients of the
approximation, :math:`\{\phi_i\}_{i = 1}^{N_n}` is the set of basis
functions for :math:`\mathcal{F}^h`, and :math:`N_n` is the
cardinality of :math:`\mathcal{F}^h`. We substitute approximations for
all functions in the weak form, resulting in a system of ordinary
differential equations (ODEs) in which we will solve for
:math:`\hat{\mathbf{u}}_i`. We will write this system of ODEs as

.. math::

   \dot{u} = v \\
   M\dot{v} + C(v) + R(u,v,p) = F_b(t) + F_q(u,t) \\
   G(u, v, p) = 0,

where each of these terms is the approximation of the terms given in
the general variational form provided above. FEniCS Mechanics parses
a user-provided dictionary to determine which terms in the above
system of ODEs need to be computed and how.


.. _finite-difference-label:

Finite Difference Methods
-------------------------

Once spatial discretization has been achieved with the FEM, we must
discretize the time derivatives in the system of ODEs given above. The
time integrators implemented in FEniCS Mechanics are currently
single-step methods.

.. _first-order-odes-label:

First Order ODEs
****************

Consider the system of ODEs

.. math::

   \dot{y} = f(y, t),

where :math:`y \in \mathbb{R}^N` for some :math:`N\in\mathbb{N}`, and
:math:`t \in \mathcal{T} \in \mathbb{R}`. A general single-step,
single-stage numerical method for approximating the solution to such a
differential equation takes the form

.. math::

   \frac{y_{n+1} - y_n}{\Delta t} = \theta f(y_{n+1}, t_{n+1}) +
         \left(1 - \theta\right)f(y_n, t_n)

Applying the generalized :math:`\theta`-method to the system of ODEs,
we get

.. math::

   f_1(u_{n+1},v_{n+1}, p_{n+1}) = 0, \\
   f_2(u_{n+1},v_{n+1}, p_{n+1}) = 0,

and

.. math::

   f_3(u_{n+1}, v_{n+1}, p_{n+1}) = 0,

where

.. math::

   f_1(u_{n+1}, v_{n+1}, p_{n+1}) = u_{n+1} - \theta\Delta t v_{n+1}
         - u_n - \Delta(1 - \theta)v_n,

.. math::

   f_2(u_{n+1},v_{n+1}) = Mv_{n+1} + \theta\Delta t\left[C(v_{n+1})
         + R(u_{n+1},v_{n+1}, p_{n+1}) - F_b(t_{n+1})
         - F_q(u_{n+1},t_{n+1})\right] \\
         - Mv_n
         + \Delta t(1-\theta)\left[C(v_n)  + R(u_n,v_n, p_n)
         - F_b(t_n) - F_q(u_n,t_n)\right],

and :math:`f_3` is the discrete variational form of the
incompressibility constraint. This discretization is used by
:class:`MechanicsProblem` and :class:`MechanicsBlockSolver`, which is
only available when FEniCS is installed with python 2, and CBC-Block
is also installed. It is also used by :class:`FluidMechanicsProblem`
and :class:`FluidMechanicsSolver` without the dependency on
displacement, :math:`u_n`, and without :math:`f_1`.


Second Order ODEs
*****************

Alternatively, one can discretize a system of second-order
ODEs. First, we substitute :math:`v = \dot{u}` into the ODE that
results from the balance of linear momentum. This gives

.. math::

   M\ddot{u} + R(u, p) = F_b(t) + F_q(u,t),

and

.. math::

   G(u, p) = 0,

where we also made use of the fact that solid mechanics problems are
formulated with respect to the reference configuration. For this
problem, we use the Newmark scheme to discretize the second-order time
derivative. The Newmark scheme is given by

.. math::

   \dot{u}_{n+1} = \dot{u}_n + \Delta t\left[(1 - \gamma)\ddot{u}_n
         + \gamma\ddot{u}_{n+1}\right],

and

.. math::

   u_{n+1} = u_n + \Delta t\dot{u}_n + \frac{1}{2}(\Delta t)^2
         \left[(1 - 2\beta)\ddot{u}_n + 2\beta\ddot{u}_{n+1}\right].

We then solve for :math:`\ddot{u}_{n+1}` in the above equations, and
substitute into the ODE given above. Then we solve the system for
:math:`u_{n+1}` (with a nonlinear solver if necessary).
