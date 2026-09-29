"""Domain-specific exceptions with user-actionable messages."""


class SimulationError(Exception):
    """Base exception for invalid simulation input or state."""


class InputValidationError(SimulationError):
    """Raised when an input file or configured scenario is invalid."""


class PlottingError(SimulationError):
    """Raised when a completed simulation result cannot be rendered as plots."""
