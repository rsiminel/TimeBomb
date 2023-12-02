# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:16:22 2022

@author: Remy
"""

# Imports
import numpy as np
from random import randint


def Fact(x):
    factorial = 1
    if x < 0:
        factorial = -1
    for i in range(1, int(x) + 1):
        factorial *= i
    return factorial


def C(k, n):
    if n < k:
        return 0
    return Fact(n) / (Fact(k) * Fact(n - k))


def A(k, n):
    if n < k:
        return 0
    return Fact(n) / Fact(n - k)


def Lklhd(n, m, k, p):
    return C(p, k) * A(p, m) * A(k - p, n - m) / A(k, n)


def Lklhd2(n, m, k, p):
    return C(p, k) * C(m - p, n - k) / C(m, n)


def IsProbability(prob):
    is_prob = True
    for i in prob:
        if prob < 0. or prob > 1.:
            is_prob = False
    return is_prob


def DistributeWires(num_players, active_wires, hand_size):
    wires = np.zeros(num_players)
    given = 0
    while given < active_wires:
        randy = randint(0, num_players - 1)
        if wires[randy] < hand_size:
            wires[randy] += 1
            given += 1
    return wires
