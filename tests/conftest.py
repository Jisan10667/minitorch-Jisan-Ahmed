"""Pytest configuration for MiniTorch."""

import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "task0_1: Task 0.1 operators")
    config.addinivalue_line("markers", "task0_2: Task 0.2 property tests") 
    config.addinivalue_line("markers", "task0_3: Task 0.3 functional")
    config.addinivalue_line("markers", "task0_4: Task 0.4 modules")
