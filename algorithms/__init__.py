# -*- coding: utf-8 -*-
"""
Moduł algorytmów dla problemu komiwojażera (TSP).

Zawiera implementacje:
- NN  - Nearest Neighbor (najbliższy sąsiad)
- IHC - Iterative Hill Climbing (wspinaczka z multistartem)
- SA  - Simulated Annealing (symulowane wyżarzanie)
- TS  - Tabu Search (przeszukiwanie tabu)
- GA  - Genetic Algorithm (algorytm genetyczny)
- ACO - Ant Colony Optimization (algorytm mrówkowy)
"""

from algorithms.nn import nearest_neighbor
from algorithms.ihc import iterative_hill_climbing, ihc_with_intensification
from algorithms.sa import simulated_annealing, sa_with_reheating
from algorithms.ts import tabu_search, tabu_search_diversification
from algorithms.ga import genetic_algorithm, ga_adaptive_mutation
from algorithms.aco import ant_colony_optimization, max_min_ant_system, aco_with_local_search

__all__ = [
    'nearest_neighbor',
    'iterative_hill_climbing',
    'ihc_with_intensification',
    'simulated_annealing',
    'sa_with_reheating',
    'tabu_search',
    'tabu_search_diversification',
    'genetic_algorithm',
    'ga_adaptive_mutation',
    'ant_colony_optimization',
    'max_min_ant_system',
    'aco_with_local_search',
]
