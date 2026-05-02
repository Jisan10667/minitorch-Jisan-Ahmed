"""
Dataset generators for classification experiments.

You'll implement documentation for these functions in Task 0.5.
"""

import random
from typing import List, Tuple


def simple(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Simple linear dataset - points on left vs right.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: x_coord >= 0.5 -> class 1, else class 0
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(1 if x >= 0.5 else 0)
    return points, labels


def diag(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Diagonal dataset - points above vs below main diagonal.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: x + y >= 1.0 -> class 1, else class 0
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(1 if x + y >= 1.0 else 0)
    return points, labels


def split(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Split dataset - points in center vs edges.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: 0.2 <= x <= 0.8 -> class 0, else class 1
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(0 if 0.2 <= x <= 0.8 else 1)
    return points, labels


def xor(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    XOR dataset - requires non-linear separation.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule:
        (x < 0.5 and y < 0.5) or (x >= 0.5 and y >= 0.5) -> class 0
        else -> class 1
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        same_side = (x < 0.5) == (y < 0.5)
        labels.append(0 if same_side else 1)
    return points, labels
