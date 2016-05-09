"""
Provides different ways to meassure a distance
"""

__author__ = 'Miguel Rivero'


import math


def euclidean_distance(vector1, vector2):
    """
    euclidean_distance between 2 vector with equal length
    >>> euclidean_distance([2,1],[5,5])
    5.0
    >>> euclidean_distance((3,4,2,6,7),(17,34,2,-4,0))
    35.2845575287547
    """
    z = 0
    for (v1, v2) in zip(vector1, vector2):
        z += (v1 - v2) ** 2
    return math.sqrt(z)


def abs_distance(x, y):
    """
    Absolute distance between 2 values

    >>> abs_distance([-3.0], [4.0])
    7.0
    """
    return abs(x[0] - y[0])