# -*- coding: utf-8 -*-
"""
Projekt IO - Problem Komiwojażera (TSP)
======================================

Implementacja i porównanie algorytmów:
1. NN  - Nearest Neighbor (najbliższy sąsiad)
2. IHC - Iterative Hill Climbing (wspinaczka z multistartem)
3. SA  - Simulated Annealing (symulowane wyżarzanie)
4. TS  - Tabu Search (przeszukiwanie tabu)
5. GA  - Genetic Algorithm (algorytm genetyczny)
6. ACO - Ant Colony Optimization (algorytm mrówkowy)

Każdy algorytm ma zaimplementowane 3 rodzaje sąsiedztw:
- swap: zamiana dwóch miast
- insert: przeniesienie miasta
- two_opt: odwrócenie fragmentu trasy

Usprawnienia:
1. IHC z intensyfikacją (autorskie)
2. SA z reheating
3. TS z dywersyfikacją (autorskie)
4. GA z adaptacyjną mutacją

Uruchomienie:
    python main.py           # Szybki test wszystkich algorytmów
    python main.py --full    # Pełne testy z wieloma parametrami
"""
import os
import sys
import time
from utils import loader, tsp
from algorithms import nn, ihc, sa, ts, ga
from algorithms import aco


def quick_test(problem, label):
    """
    Szybki test wszystkich algorytmów z domyślnymi parametrami.
    
    Uruchamia każdy z 6 algorytmów raz i porównuje wyniki.
    Przydatne do szybkiego sprawdzenia czy wszystko działa.
    
    Args:
        problem: obiekt TSP z macierzą odległości
        label: nazwa instancji (np. "TSP_48")
    """
    print(f"\n{'='*60}")
    print(f"INSTANCJA: {label} | Liczba miast: {problem.n}")
    print(f"{'='*60}")
    
    results = {}  # Słownik: algorytm -> (koszt, czas)
    
    # === 1. NN - Nearest Neighbor (najbliższy sąsiad) ===
    # Prosty algorytm zachłanny - szybki ale nieoptymaly
    start_time = time.perf_counter()
    route, cost = nn.nearest_neighbor(problem, start=0)
    elapsed = time.perf_counter() - start_time
    results['NN'] = (cost, elapsed)
    print(f"NN:  {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # === 2. IHC - Iterative Hill Climbing (wspinaczka) ===
    # Multistart + akceptacja tylko lepszych rozwiązań
    start_time = time.perf_counter()
    route, cost = ihc.iterative_hill_climbing(
        problem, iterations=1000, restarts=10, neighborhood="two_opt"
    )
    elapsed = time.perf_counter() - start_time
    results['IHC'] = (cost, elapsed)
    print(f"IHC: {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # === 3. SA - Simulated Annealing (wyżarzanie) ===
    # Akceptuje gorsze rozwiązania z malejącym prawdopodobieństwem
    start_time = time.perf_counter()
    route, cost = sa.simulated_annealing(
        problem, temp=1000, alpha=0.99, iterations=5000, neighborhood="two_opt"
    )
    elapsed = time.perf_counter() - start_time
    results['SA'] = (cost, elapsed)
    print(f"SA:  {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # === 4. TS - Tabu Search (przeszukiwanie tabu) ===
    # Zabrania powrotu do ostatnio odwiedzonych rozwiązań
    start_time = time.perf_counter()
    route, cost = ts.tabu_search(
        problem, iterations=500, tabu_size=20, neighborhood="two_opt"
    )
    elapsed = time.perf_counter() - start_time
    results['TS'] = (cost, elapsed)
    print(f"TS:  {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # === 5. GA - Genetic Algorithm (genetyczny) ===
    # Populacja ewoluuje przez selekcję, krzyżowanie, mutację
    start_time = time.perf_counter()
    route, cost = ga.genetic_algorithm(
        problem, pop_size=100, generations=100,
        selection_type="tournament", crossover_type="ox"
    )
    elapsed = time.perf_counter() - start_time
    results['GA'] = (cost, elapsed)
    print(f"GA:  {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # === 6. ACO - Ant Colony Optimization (mrówkowy) ===
    # Kolonia mrówek zostawia feromony na krótszych ścieżkach
    start_time = time.perf_counter()
    route, cost = aco.ant_colony_optimization(
        problem, n_ants=20, n_iterations=50
    )
    elapsed = time.perf_counter() - start_time
    results['ACO'] = (cost, elapsed)
    print(f"ACO: {cost:12.2f} | czas: {elapsed:.4f}s")
    
    # Pokaż najlepszy wynik
    best_alg = min(results, key=lambda x: results[x][0])
    print(f"\nNajlepszy: {best_alg} = {results[best_alg][0]:.2f}")
    
    return results


def run_instance(label, path, full_test=False, n_runs=5):
    """
    Uruchamia testy dla danej instancji.
    """
    if not os.path.exists(path):
        print(f"Błąd: Nie znaleziono pliku: {path}")
        return None
    
    try:
        coords = loader.load_tsp_file(path)
        problem = tsp.TSP(coords)
        
        if full_test:
            from experiments.run_tests import run_all_tests
            return run_all_tests(problem, label, n_runs=n_runs)
        else:
            return quick_test(problem, label)
    
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Główna funkcja programu.
    """
    print("="*60)
    print("PROJEKT IO - PROBLEM KOMIWOJAŻERA (TSP)")
    print("="*60)
    
    # Sprawdź argumenty
    full_test = "--full" in sys.argv
    n_runs = 5
    
    # Możliwość zmiany liczby powtórzeń
    for arg in sys.argv:
        if arg.startswith("--runs="):
            try:
                n_runs = int(arg.split("=")[1])
            except:
                pass
    
    if full_test:
        print(f"TRYB: Pełne testy z {n_runs} powtórzeniami")
        print("UWAGA: To może zająć kilka minut...")
    else:
        print("TRYB: Szybki test (użyj --full dla pełnych testów)")
    
    # Ścieżki do instancji
    instances = [
        ("TSP_48", "instances/Dane_TSP_48.tsp"),
        ("TSP_76", "instances/Dane_TSP_76.tsp"),
        ("TSP_127", "instances/Dane_TSP_127.tsp"),
    ]
    
    all_results = {}
    
    for label, path in instances:
        result = run_instance(label, path, full_test=full_test, n_runs=n_runs)
        if result:
            all_results[label] = result
    
    # Podsumowanie końcowe
    print("\n" + "="*60)
    print("PODSUMOWANIE KOŃCOWE")
    print("="*60)
    
    if not full_test and all_results:
        print(f"\n{'Instancja':<12} | {'Najlepszy':<8} | {'Koszt':>12}")
        print("-" * 40)
        for label, results in all_results.items():
            if isinstance(results, dict):
                best_alg = min(results, key=lambda x: results[x][0])
                best_cost = results[best_alg][0]
                print(f"{label:<12} | {best_alg:<8} | {best_cost:>12.2f}")
    
    print("\n>>> KONIEC PROGRAMU <<<")


if __name__ == "__main__":
    main()