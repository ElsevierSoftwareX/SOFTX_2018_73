import dolfin as dlf

from .mechanicsproblem import MechanicsProblem

class MechanicsSolver:
    """


    """

    def __init__(self, mechanicsProblem):

        # Check if unsteady simulation
        # If simulation is unsteady, the following must be
        # updated at each step:
        #
        # - solution at current step
        # - solution at previous step
        # - convective acceleration (if present)
        # - stiffness matrix
        #   - geometric stiffness (might already be taken care of by dlf.derivative)
        #   - material stiffness (might already be taken care of by dlf.derivative)
        # - body force (if unsteady)
        # - traction BC (if unsteady)

        self._mechanicsProblem = mechanicsProblem

        return None
