# -*- coding: utf-8 -*-
"""
Moduł eksperymentów dla problemu komiwojażera (TSP).

Zawiera:
- run_tests: kompleksowe testy wszystkich algorytmów
- runner: pomocnicze funkcje do uruchamiania testów
"""

from experiments.run_tests import (
    run_all_tests,
    run_multiple_times,
    test_nn,
    test_ihc,
    test_sa,
    test_ts,
    test_ga,
    test_aco,
)

__all__ = [
    'run_all_tests',
    'run_multiple_times',
    'test_nn',
    'test_ihc',
    'test_sa',
    'test_ts',
    'test_ga',
    'test_aco',
]
