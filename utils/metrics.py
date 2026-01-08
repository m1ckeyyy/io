# -*- coding: utf-8 -*-
"""
Stary plik z sąsiedztwami - używaj neighborhoods.py zamiast tego.
Ten plik pozostał dla kompatybilności wstecznej.
"""
import random


def swap(route):
    """Zamiana dwóch miast miejscami."""
    a, b = random.sample(range(len(route)), 2)
    new = route.copy()
    new[a], new[b] = new[b], new[a]
    return new

def insert(route):
    a, b = random.sample(range(len(route)), 2)
    new = route.copy()
    city = new.pop(a)
    new.insert(b, city)
    return new

def two_opt(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    new = route[:]
    new[a:b] = reversed(new[a:b])
    return new

# ===== FAST / UJEDNOLICONE =====
def two_opt_delta(route, tsp):
    n = len(route)
    a, b = sorted(random.sample(range(n), 2))

    if b - a < 2:
        return route, 0.0

    A = route[a - 1]
    B = route[a]
    C = route[b - 1]
    D = route[b % n]

    before = tsp.dist_matrix[A][B] + tsp.dist_matrix[C][D]
    after = tsp.dist_matrix[A][C] + tsp.dist_matrix[B][D]

    new_route = route[:]
    new_route[a:b] = reversed(new_route[a:b])

    return new_route, after - before
