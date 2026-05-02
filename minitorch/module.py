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
        # Initialize in Task 0.4
        self._modules: Dict[str, "Module"] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training: bool = True
    
    def modules(self) -> Sequence["Module"]:
        """Return all sub-modules."""
        m: List["Module"] = list(self.__dict__["_modules"].values())
        return m
    
    def train(self):
        """Set training mode."""
        self.training = True
        for m in self.modules():
            m.train()
    
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        for m in self.modules():
            m.eval()
    
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Return all parameters with names."""
        res: List[Tuple[str, Parameter]] = []
        for k, v in self.__dict__["_parameters"].items():
            res.append((k, v))
        for mod_name, m in self.__dict__["_modules"].items():
            for k, v in m.named_parameters():
                res.append((f"{mod_name}.{k}", v))
        return res
    
    def parameters(self) -> Sequence[Parameter]:
        """Return all parameters."""
        return [p for _, p in self.named_parameters()]
    
    def add_parameter(self, name: str, value: Any) -> Parameter:
        """Add a parameter."""
        p = Parameter(value)
        self.__dict__["_parameters"][name] = p
        return p
    
    def __setattr__(self, key: str, value: Any):
        """Custom attribute setter."""
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][key] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][key] = value
        super().__setattr__(key, value)
