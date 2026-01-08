# -*- coding: utf-8 -*-
"""
Moduł narzędziowy dla problemu komiwojażera (TSP).

Zawiera:
- loader: wczytywanie plików .tsp
- tsp: klasa TSP z macierzą odległości
- neighborhoods: funkcje sąsiedztwa (swap, insert, two_opt)
- metrics: metryki i funkcje pomocnicze
"""

from utils.loader import load_tsp_file
from utils.tsp import TSP
from utils.neighborhoods import (
    swap, insert, two_opt,
    swap_delta, insert_delta, two_opt_delta,
    NEIGHBORHOODS, NEIGHBORHOODS_DELTA
)

__all__ = [
    'load_tsp_file',
    'TSP',
    'swap', 'insert', 'two_opt',
    'swap_delta', 'insert_delta', 'two_opt_delta',
    'NEIGHBORHOODS', 'NEIGHBORHOODS_DELTA',
]
