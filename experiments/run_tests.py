# -*- coding: utf-8 -*-
"""
Moduł do kompleksowego testowania algorytmów TSP.

Wykonuje testy dla wszystkich algorytmów z różnymi parametrami,
każdy test powtarzany min. 5 razy, zapisuje wyniki do CSV.

Wymagania projektu:
- Min. 4 różne wartości dla każdego parametru
- Min. 5 powtórzeń dla algorytmów z losowością
- Zapisywanie wartości minimalnych i średnich
- Czas wykonania
"""
import time
import csv
import os
import statistics
from datetime import datetime

from algorithms.nn import nearest_neighbor
from algorithms.ihc import iterative_hill_climbing, ihc_with_intensification
from algorithms.sa import simulated_annealing, sa_with_reheating
from algorithms.ts import tabu_search, tabu_search_diversification
from algorithms.ga import genetic_algorithm, ga_adaptive_mutation
from algorithms.aco import ant_colony_optimization, max_min_ant_system


def run_multiple_times(func, n_runs=5):
    """
    Uruchamia funkcję wielokrotnie i zbiera statystyki.
    
    Returns:
        dict z kluczami: min, max, mean, std, times, all_results
    """
    results = []
    times = []
    routes = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        route, cost = func()
        elapsed = time.perf_counter() - start
        results.append(cost)
        times.append(elapsed)
        routes.append(route)
    
    return {
        'min': min(results),
        'max': max(results),
        'mean': statistics.mean(results),
        'std': statistics.stdev(results) if len(results) > 1 else 0,
        'mean_time': statistics.mean(times),
        'all_results': results,
        'all_times': times,
        'best_route': routes[results.index(min(results))]
    }


def test_nn(tsp, n_runs=5):
    """
    Testuje algorytm NN dla różnych miast startowych.
    NN jest deterministyczny dla danego startu, więc n_runs=1 wystarczy.
    
    NN ma tylko 1 parametr (miasto startowe), ale testujemy wiele wartości.
    Dodatkowo testujemy wariant z wyborem najlepszego startu spośród kilku.
    """
    results = []
    
    # Parametr 1: miasto startowe (testujemy min 4 różne wartości)
    # Wybieramy 4 miasta równomiernie rozłożone
    n = tsp.n
    start_cities = [0, n//4, n//2, 3*n//4]
    start_cities = [min(s, n-1) for s in start_cities]  # upewnij się że są w zakresie
    
    print("  Testowanie NN - różne miasta startowe...")
    for start in start_cities:
        start_time = time.perf_counter()
        route, cost = nearest_neighbor(tsp, start=start)
        elapsed = time.perf_counter() - start_time
        
        results.append({
            'algorithm': 'NN',
            'params': f'start={start}',
            'min': cost,
            'mean': cost,
            'std': 0,
            'time': elapsed,
            'route': route
        })
        
        print(f"    NN | start={start} | cost={cost:.2f} | time={elapsed:.4f}s")
    
    # Parametr 2: Wariant - najlepszy z K losowych startów
    # (symuluje element losowości i daje porównanie jakości)
    print("  Testowanie NN - najlepszy z K startów...")
    k_values = [1, 5, 10, 20]
    
    for k in k_values:
        start_time = time.perf_counter()
        best_cost = float('inf')
        best_route = None
        
        # Dla małych instancji testuj wszystkie, dla dużych losowe
        test_starts = list(range(min(k, n)))
        
        for s in test_starts:
            route, cost = nearest_neighbor(tsp, start=s)
            if cost < best_cost:
                best_cost = cost
                best_route = route
        
        elapsed = time.perf_counter() - start_time
        
        results.append({
            'algorithm': 'NN',
            'params': f'best_of_k={k}',
            'min': best_cost,
            'mean': best_cost,
            'std': 0,
            'time': elapsed,
            'route': best_route
        })
        
        print(f"    NN | best_of_k={k} | cost={best_cost:.2f} | time={elapsed:.4f}s")
    
    return results


def test_ihc(tsp, n_runs=5):
    """
    Testuje algorytm IHC z różnymi parametrami.
    
    Parametry testowane (min. 4 dla grupy 4-osobowej):
    1. Rodzaj sąsiedztwa (3 wartości)
    2. Liczba iteracji (4 wartości)
    3. Liczba restartów (4 wartości)
    4. Limit iteracji bez poprawy (4 wartości)
    """
    results = []
    
    # Parametry do testowania
    neighborhoods = ["swap", "insert", "two_opt"]
    iterations_list = [100, 500, 1000, 2000]
    restarts_list = [5, 10, 20, 30]
    no_improve_limits = [50, 100, 200, 500]  # NOWY PARAMETR 4
    
    print("  Testowanie IHC...")
    
    # Parametr 1: Test sąsiedztw
    for neigh in neighborhoods:
        stats = run_multiple_times(
            lambda n=neigh: iterative_hill_climbing(tsp, iterations=1000, restarts=10, neighborhood=n),
            n_runs
        )
        results.append({
            'algorithm': 'IHC',
            'params': f'neigh={neigh}, iters=1000, restarts=10',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    neigh={neigh} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 2: Test liczby iteracji
    for iters in iterations_list:
        stats = run_multiple_times(
            lambda i=iters: iterative_hill_climbing(tsp, iterations=i, restarts=10, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'IHC',
            'params': f'neigh=two_opt, iters={iters}, restarts=10',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    iters={iters} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 3: Test liczby restartów
    for restarts in restarts_list:
        stats = run_multiple_times(
            lambda r=restarts: iterative_hill_climbing(tsp, iterations=1000, restarts=r, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'IHC',
            'params': f'neigh=two_opt, iters=1000, restarts={restarts}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    restarts={restarts} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 4: Test limitu iteracji bez poprawy
    for no_imp in no_improve_limits:
        stats = run_multiple_times(
            lambda ni=no_imp: iterative_hill_climbing(tsp, iterations=2000, restarts=10, 
                                                      neighborhood="two_opt", no_improve_limit=ni),
            n_runs
        )
        results.append({
            'algorithm': 'IHC',
            'params': f'neigh=two_opt, iters=2000, no_improve={no_imp}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    no_improve_limit={no_imp} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test usprawnienia: IHC z intensyfikacją
    stats = run_multiple_times(
        lambda: ihc_with_intensification(tsp, iterations=1000, restarts=10),
        n_runs
    )
    results.append({
        'algorithm': 'IHC_INTENSIFICATION',
        'params': 'iters=1000, restarts=10',
        'min': stats['min'],
        'mean': stats['mean'],
        'std': stats['std'],
        'time': stats['mean_time'],
        'route': stats['best_route']
    })
    print(f"    IHC+Intensification | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    return results


def test_sa(tsp, n_runs=5):
    """
    Testuje algorytm SA z różnymi parametrami.
    """
    results = []
    
    # Parametry do testowania
    neighborhoods = ["swap", "insert", "two_opt"]
    temps = [100, 500, 1000, 5000]
    alphas = [0.9, 0.95, 0.99, 0.995]
    cooling_methods = ["geometric", "linear", "logarithmic"]
    
    print("  Testowanie SA...")
    
    # Test sąsiedztw
    for neigh in neighborhoods:
        stats = run_multiple_times(
            lambda n=neigh: simulated_annealing(tsp, temp=1000, alpha=0.99, iterations=5000, neighborhood=n),
            n_runs
        )
        results.append({
            'algorithm': 'SA',
            'params': f'neigh={neigh}, temp=1000, alpha=0.99, iters=5000',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    neigh={neigh} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test temperatury początkowej
    for temp in temps:
        stats = run_multiple_times(
            lambda t=temp: simulated_annealing(tsp, temp=t, alpha=0.99, iterations=5000, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'SA',
            'params': f'neigh=two_opt, temp={temp}, alpha=0.99, iters=5000',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    temp={temp} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test współczynnika chłodzenia
    for alpha in alphas:
        stats = run_multiple_times(
            lambda a=alpha: simulated_annealing(tsp, temp=1000, alpha=a, iterations=5000, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'SA',
            'params': f'neigh=two_opt, temp=1000, alpha={alpha}, iters=5000',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    alpha={alpha} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test metod chłodzenia
    for method in cooling_methods:
        stats = run_multiple_times(
            lambda m=method: simulated_annealing(tsp, temp=1000, alpha=0.99, iterations=5000, 
                                                  neighborhood="two_opt", cooling_method=m),
            n_runs
        )
        results.append({
            'algorithm': 'SA',
            'params': f'neigh=two_opt, temp=1000, cooling={method}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    cooling={method} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test usprawnienia: SA z reheating
    stats = run_multiple_times(
        lambda: sa_with_reheating(tsp, temp=1000, alpha=0.99, iterations=5000),
        n_runs
    )
    results.append({
        'algorithm': 'SA_REHEATING',
        'params': 'temp=1000, alpha=0.99, iters=5000',
        'min': stats['min'],
        'mean': stats['mean'],
        'std': stats['std'],
        'time': stats['mean_time'],
        'route': stats['best_route']
    })
    print(f"    SA+Reheating | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    return results


def test_ts(tsp, n_runs=5):
    """
    Testuje algorytm TS z różnymi parametrami.
    
    Parametry testowane (min. 4 dla grupy 4-osobowej):
    1. Rodzaj sąsiedztwa (3 wartości)
    2. Długość listy tabu (4 wartości)
    3. Liczba iteracji (4 wartości)  
    4. Liczba kandydatów na iterację (4 wartości)
    """
    results = []
    
    # Parametry do testowania
    neighborhoods = ["swap", "insert", "two_opt"]
    tabu_sizes = [5, 10, 20, 50]
    iterations_list = [100, 250, 500, 1000]
    candidates_list = [5, 10, 20, 40]  # NOWY PARAMETR 4
    
    print("  Testowanie TS...")
    
    # Parametr 1: Test sąsiedztw
    for neigh in neighborhoods:
        stats = run_multiple_times(
            lambda n=neigh: tabu_search(tsp, iterations=500, tabu_size=20, neighborhood=n),
            n_runs
        )
        results.append({
            'algorithm': 'TS',
            'params': f'neigh={neigh}, iters=500, tabu_size=20',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    neigh={neigh} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 2: Test długości listy tabu
    for tabu_size in tabu_sizes:
        stats = run_multiple_times(
            lambda ts=tabu_size: tabu_search(tsp, iterations=500, tabu_size=ts, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'TS',
            'params': f'neigh=two_opt, iters=500, tabu_size={tabu_size}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    tabu_size={tabu_size} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 3: Test liczby iteracji
    for iters in iterations_list:
        stats = run_multiple_times(
            lambda i=iters: tabu_search(tsp, iterations=i, tabu_size=20, neighborhood="two_opt"),
            n_runs
        )
        results.append({
            'algorithm': 'TS',
            'params': f'neigh=two_opt, iters={iters}, tabu_size=20',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    iters={iters} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Parametr 4: Test liczby kandydatów na iterację
    for cand in candidates_list:
        stats = run_multiple_times(
            lambda c=cand: tabu_search(tsp, iterations=500, tabu_size=20, neighborhood="two_opt", 
                                       candidates_per_iter=c),
            n_runs
        )
        results.append({
            'algorithm': 'TS',
            'params': f'neigh=two_opt, iters=500, candidates={cand}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    candidates={cand} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test usprawnienia: TS z dywersyfikacją
    stats = run_multiple_times(
        lambda: tabu_search_diversification(tsp, iterations=500, tabu_size=20),
        n_runs
    )
    results.append({
        'algorithm': 'TS_DIVERSIFICATION',
        'params': 'iters=500, tabu_size=20',
        'min': stats['min'],
        'mean': stats['mean'],
        'std': stats['std'],
        'time': stats['mean_time'],
        'route': stats['best_route']
    })
    print(f"    TS+Diversification | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    return results


def test_ga(tsp, n_runs=5):
    """
    Testuje algorytm GA z różnymi parametrami.
    """
    results = []
    
    # Parametry do testowania
    selection_types = ["tournament", "roulette", "ranking"]
    crossover_types = ["ox", "pmx", "cx"]
    mutation_types = ["swap", "insert", "inversion"]
    pop_sizes = [50, 100, 150, 200]
    mutation_probs = [0.01, 0.05, 0.1, 0.2]
    
    print("  Testowanie GA...")
    
    # Test metod selekcji
    for sel in selection_types:
        stats = run_multiple_times(
            lambda s=sel: genetic_algorithm(tsp, pop_size=100, generations=100, 
                                            selection_type=s, crossover_type="ox"),
            n_runs
        )
        results.append({
            'algorithm': 'GA',
            'params': f'sel={sel}, cross=ox, pop=100, gen=100',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    selection={sel} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test metod krzyżowania
    for cross in crossover_types:
        stats = run_multiple_times(
            lambda c=cross: genetic_algorithm(tsp, pop_size=100, generations=100,
                                              selection_type="tournament", crossover_type=c),
            n_runs
        )
        results.append({
            'algorithm': 'GA',
            'params': f'sel=tournament, cross={cross}, pop=100, gen=100',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    crossover={cross} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test rodzajów mutacji
    for mut in mutation_types:
        stats = run_multiple_times(
            lambda m=mut: genetic_algorithm(tsp, pop_size=100, generations=100,
                                            selection_type="tournament", crossover_type="ox",
                                            mutation_type=m),
            n_runs
        )
        results.append({
            'algorithm': 'GA',
            'params': f'sel=tournament, cross=ox, mut={mut}, pop=100, gen=100',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    mutation={mut} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test wielkości populacji
    for pop in pop_sizes:
        stats = run_multiple_times(
            lambda p=pop: genetic_algorithm(tsp, pop_size=p, generations=100,
                                            selection_type="tournament", crossover_type="ox"),
            n_runs
        )
        results.append({
            'algorithm': 'GA',
            'params': f'sel=tournament, cross=ox, pop={pop}, gen=100',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    pop_size={pop} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test prawdopodobieństwa mutacji
    for p_mut in mutation_probs:
        stats = run_multiple_times(
            lambda pm=p_mut: genetic_algorithm(tsp, pop_size=100, generations=100,
                                               selection_type="tournament", crossover_type="ox",
                                               p_mut=pm),
            n_runs
        )
        results.append({
            'algorithm': 'GA',
            'params': f'sel=tournament, cross=ox, p_mut={p_mut}, pop=100, gen=100',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    p_mut={p_mut} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test usprawnienia: GA z adaptacyjną mutacją
    stats = run_multiple_times(
        lambda: ga_adaptive_mutation(tsp, pop_size=100, generations=100),
        n_runs
    )
    results.append({
        'algorithm': 'GA_ADAPTIVE',
        'params': 'pop=100, gen=100, adaptive_mutation',
        'min': stats['min'],
        'mean': stats['mean'],
        'std': stats['std'],
        'time': stats['mean_time'],
        'route': stats['best_route']
    })
    print(f"    GA+AdaptiveMutation | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    return results


def test_aco(tsp, n_runs=5):
    """
    Testuje algorytm ACO z różnymi parametrami.
    """
    results = []
    
    # Parametry do testowania
    n_ants_list = [10, 20, 30, 50]
    alphas = [0.5, 1.0, 2.0, 3.0]
    betas = [1.0, 2.0, 3.0, 5.0]
    rhos = [0.1, 0.3, 0.5, 0.7]
    
    print("  Testowanie ACO...")
    
    # Test liczby mrówek
    for n_ants in n_ants_list:
        stats = run_multiple_times(
            lambda na=n_ants: ant_colony_optimization(tsp, n_ants=na, n_iterations=50),
            n_runs
        )
        results.append({
            'algorithm': 'ACO',
            'params': f'n_ants={n_ants}, iters=50, alpha=1, beta=2',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    n_ants={n_ants} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test parametru alpha (wpływ feromonów)
    for alpha in alphas:
        stats = run_multiple_times(
            lambda a=alpha: ant_colony_optimization(tsp, n_ants=20, n_iterations=50, alpha=a),
            n_runs
        )
        results.append({
            'algorithm': 'ACO',
            'params': f'n_ants=20, iters=50, alpha={alpha}, beta=2',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    alpha={alpha} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test parametru beta (wpływ heurystyki)
    for beta in betas:
        stats = run_multiple_times(
            lambda b=beta: ant_colony_optimization(tsp, n_ants=20, n_iterations=50, beta=b),
            n_runs
        )
        results.append({
            'algorithm': 'ACO',
            'params': f'n_ants=20, iters=50, alpha=1, beta={beta}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    beta={beta} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test parametru rho (parowanie)
    for rho in rhos:
        stats = run_multiple_times(
            lambda r=rho: ant_colony_optimization(tsp, n_ants=20, n_iterations=50, rho=r),
            n_runs
        )
        results.append({
            'algorithm': 'ACO',
            'params': f'n_ants=20, iters=50, rho={rho}',
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': stats['mean_time'],
            'route': stats['best_route']
        })
        print(f"    rho={rho} | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    # Test wariantu MMAS
    stats = run_multiple_times(
        lambda: max_min_ant_system(tsp, n_ants=20, n_iterations=50),
        n_runs
    )
    results.append({
        'algorithm': 'MMAS',
        'params': 'n_ants=20, iters=50 (MAX-MIN)',
        'min': stats['min'],
        'mean': stats['mean'],
        'std': stats['std'],
        'time': stats['mean_time'],
        'route': stats['best_route']
    })
    print(f"    MMAS | min={stats['min']:.2f} | mean={stats['mean']:.2f}")
    
    return results


def run_all_tests(tsp, instance_name, n_runs=5, output_dir="results"):
    """
    Uruchamia wszystkie testy dla danej instancji.
    """
    print(f"\n{'='*60}")
    print(f"TESTOWANIE INSTANCJI: {instance_name}")
    print(f"Liczba miast: {tsp.n}")
    print(f"Liczba powtórzeń: {n_runs}")
    print(f"{'='*60}")
    
    all_results = []
    
    # Uruchom testy dla każdego algorytmu
    print("\n[1/6] Algorytm NN (Nearest Neighbor)")
    all_results.extend(test_nn(tsp, n_runs=1))  # NN deterministyczny
    
    print("\n[2/6] Algorytm IHC (Iterative Hill Climbing)")
    all_results.extend(test_ihc(tsp, n_runs))
    
    print("\n[3/6] Algorytm SA (Simulated Annealing)")
    all_results.extend(test_sa(tsp, n_runs))
    
    print("\n[4/6] Algorytm TS (Tabu Search)")
    all_results.extend(test_ts(tsp, n_runs))
    
    print("\n[5/6] Algorytm GA (Genetic Algorithm)")
    all_results.extend(test_ga(tsp, n_runs))
    
    print("\n[6/6] Algorytm ACO (Ant Colony Optimization)")
    all_results.extend(test_aco(tsp, n_runs))
    
    # Zapisz wyniki do CSV
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"results_{instance_name}_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['algorithm', 'params', 'min', 'mean', 'std', 'time'])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k != 'route'})
    
    print(f"\nWyniki zapisane do: {csv_path}")
    
    # Podsumowanie - najlepsze wyniki
    print(f"\n{'='*60}")
    print("NAJLEPSZE WYNIKI DLA KAŻDEGO ALGORYTMU:")
    print(f"{'='*60}")
    
    algorithms = set(r['algorithm'] for r in all_results)
    best_overall = None
    best_overall_cost = float('inf')
    
    for alg in sorted(algorithms):
        alg_results = [r for r in all_results if r['algorithm'] == alg]
        best = min(alg_results, key=lambda x: x['min'])
        print(f"{alg:20s} | min={best['min']:12.2f} | params: {best['params']}")
        
        if best['min'] < best_overall_cost:
            best_overall_cost = best['min']
            best_overall = best
    
    print(f"\n{'='*60}")
    print(f"NAJLEPSZY WYNIK OGÓLNY: {best_overall['algorithm']} = {best_overall['min']:.2f}")
    print(f"Parametry: {best_overall['params']}")
    print(f"{'='*60}")
    
    return all_results, best_overall

