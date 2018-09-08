__all__ = ["InvalidOption", "InvalidCombination", "InconsistentCombination",
           "DimensionMismatch", "RequiredParameter"]

class InvalidOption(ValueError):
    pass
    # def __init__(self, message):
    #     super().__init__(message)

class InvalidCombination(InvalidOption):
    pass
    # def __init__(self, message):
    #     super().__init__(message)

class InconsistentCombination(Exception):
    pass

class DimensionMismatch(ValueError):
    pass

class RequiredParameter(Exception):
    pass
