"""
This module defines Exception classes specific to FEniCS Mechanics.
"""
__all__ = ["InvalidOption", "InvalidCombination", "InconsistentCombination",
           "DimensionMismatch", "RequiredParameter"]

# Base class to allow catching any exception raised within FEniCS Mechanics.
# Current code does not make use of this.
class FEniCSMechanicsError(Exception):
    pass

class InvalidOption(FEniCSMechanicsError, ValueError):
    pass

class InvalidCombination(InvalidOption):
    pass

class InconsistentCombination(FEniCSMechanicsError):
    pass

class DimensionMismatch(FEniCSMechanicsError):
    pass

class RequiredParameter(FEniCSMechanicsError):
    pass

class SoftwareNotAvailable(FEniCSMechanicsError):
    pass
