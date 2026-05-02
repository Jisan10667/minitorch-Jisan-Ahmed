"""
Module system for organizing neural network components and parameters.

This provides the foundation for building complex neural networks
with proper parameter management and hierarchy.
"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A trainable parameter in a neural network."""
    
    def __init__(self, value: Any):
        self.value = value
        
    @property
    def shape(self):
        """Return parameter shape."""
        if hasattr(self.value, 'shape'):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Parameter(value={self.value})"


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        # TODO: Initialize in Task 0.4
        raise NotImplementedError("Implement in Task 0.4")
    
    def modules(self) -> Sequence["Module"]:
        """Return all sub-modules."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def train(self):
        """Set training mode."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def eval(self):
        """Set evaluation mode."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Return all parameters with names."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def parameters(self) -> Sequence[Parameter]:
        """Return all parameters."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def add_parameter(self, name: str, value: Any) -> Parameter:
        """Add a parameter."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def __setattr__(self, key: str, value: Any):
        """Custom attribute setter."""
        raise NotImplementedError("Implement in Task 0.4")
