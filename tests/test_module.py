"""Tests for module system."""

import pytest
from minitorch.module import Module, Parameter

@pytest.mark.task0_4
def test_module_init():
    module = Module()
    assert module.training == True

@pytest.mark.task0_4
def test_train_eval():
    module = Module()
    assert module.training == True
    
    module.eval()
    assert module.training == False
    
    module.train()
    assert module.training == True

@pytest.mark.task0_4
def test_parameter():
    param = Parameter(5.0)
    assert param.value == 5.0

@pytest.mark.task0_4
def test_add_parameter():
    module = Module()
    param = module.add_parameter("weight", 3.14)
    assert isinstance(param, Parameter)
    assert param.value == 3.14

@pytest.mark.task0_4
def test_nested_modules():
    parent = Module()
    child = Module()
    parent.child = child
    
    modules = list(parent.modules())
    assert child in modules
