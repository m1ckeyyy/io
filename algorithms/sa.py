# -*- coding: utf-8 -*-
"""
Algorytm Symulowanego Wyżarzania (SA - Simulated Annealing)
dla problemu komiwojażera (TSP).

Parametry:
- temp: temperatura początkowa
- alpha: współczynnik redukcji temperatury (cooling rate)
- iterations: liczba iteracji
- neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
- cooling_method: metoda chłodzenia ("geometric", "linear", "logarithmic")
- iterations_per_temp: liczba iteracji dla każdej temperatury
"""
import math
import random
from utils.neighborhoods import NEIGHBORHOODS, NEIGHBORHOODS_DELTA


def simulated_annealing(
    tsp,
    temp=1000,
    alpha=0.99,
    iterations=5000,
    neighborhood="two_opt",
    cooling_method="geometric",
    iterations_per_temp=1,
    use_nn_start=False
):
    """
    Symulowane Wyżarzanie (SA)
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        temp: temperatura początkowa
        alpha: współczynnik chłodzenia (dla geometric: 0.9-0.99)
        iterations: maksymalna liczba iteracji
        neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
        cooling_method: "geometric", "linear", "logarithmic"
        iterations_per_temp: ile rozwiązań sprawdzić dla każdej temperatury
        use_nn_start: czy startować z rozwiązania NN
    
    Returns:
        (best_route, best_dist)
    """
    n = tsp.n
    
    # Wybór funkcji sąsiedztwa
    if neighborhood in NEIGHBORHOODS:
        move_func = NEIGHBORHOODS[neighborhood]
    else:
        move_func = NEIGHBORHOODS["two_opt"]
    
    # Generowanie rozwiązania startowego
    if use_nn_start:
        from algorithms.nn import nearest_neighbor
        current_route, current_dist = nearest_neighbor(tsp, start=random.randint(0, n-1))
        current_route = list(current_route)
    else:
        current_route = list(range(n))
        random.shuffle(current_route)
        current_dist = tsp.route_length(current_route)
    
    best_route = current_route[:]
    best_dist = current_dist
    
    t = temp
    initial_temp = temp
    
    # === PĘTLA GŁÓWNA SYMULOWANEGO WYŻARZANIA ===
    for i in range(iterations):
        
        # --- Iteracje dla aktualnej temperatury ---
        for _ in range(iterations_per_temp):
            # Generuj losowego sąsiada
            neighbor = move_func(current_route)
            neighbor_dist = tsp.route_length(neighbor)
            diff = neighbor_dist - current_dist  # Różnica kosztów
            
            # === KRYTERIUM AKCEPTACJI METROPOLIS ===
            # Kluczowy element SA - pozwala akceptować gorsze rozwiązania!
            
            if diff < 0:
                # Lepsze rozwiązanie - ZAWSZE akceptuj
                current_route = neighbor
                current_dist = neighbor_dist
                # Sprawdź czy nowe najlepsze globalne
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_route = current_route[:]
            elif t > 0:
                # Gorsze rozwiązanie - akceptuj z PRAWDOPODOBIEŃSTWEM
                # P = exp(-diff/T) - im wyższa temp, tym większa szansa
                prob = math.exp(-diff / t)
                if random.random() < prob:
                    current_route = neighbor  # Akceptuj gorsze!
                    current_dist = neighbor_dist
        
        # --- Redukcja temperatury (chłodzenie) ---
        # Temperatura maleje co iterację wg wybranego schematu
        t = _reduce_temperature(t, initial_temp, alpha, i, iterations, cooling_method)
        
        # Zatrzymaj jeśli temperatura praktycznie zerowa
        if t < 1e-10:
            break
    
    return best_route, best_dist


def _reduce_temperature(t, initial_temp, alpha, iteration, max_iterations, method):
    """
    Różne metody redukcji temperatury.
    """
    if method == "geometric":
        # T(k+1) = alpha * T(k)
        return t * alpha
    elif method == "linear":
        # T(k) = T0 - k * (T0 / max_iter)
        return initial_temp * (1 - iteration / max_iterations)
    elif method == "logarithmic":
        # T(k) = T0 / log(k + 2)
        return initial_temp / math.log(iteration + 2)
    else:
        return t * alpha


def sa_with_reheating(
    tsp,
    temp=1000,
    alpha=0.99,
    iterations=5000,
    neighborhood="two_opt",
    reheat_threshold=100,
    reheat_factor=0.5
):
    """
    USPRAWNIENIE: SA z podgrzewaniem (reheating)
    
    Gdy przez długi czas nie ma poprawy, "podgrzewamy" temperaturę
    aby umożliwić ucieczkę z minimum lokalnego.
    
    Args:
        tsp: obiekt TSP
        temp: temperatura początkowa
        alpha: współczynnik chłodzenia
        iterations: liczba iteracji
        neighborhood: typ sąsiedztwa
        reheat_threshold: liczba iteracji bez poprawy do podgrzania
        reheat_factor: jaki procent początkowej temperatury przywrócić
    
    Returns:
        (best_route, best_dist)
    """
    n = tsp.n
    move_func = NEIGHBORHOODS.get(neighborhood, NEIGHBORHOODS["two_opt"])
    
    current_route = list(range(n))
    random.shuffle(current_route)
    current_dist = tsp.route_length(current_route)
    
    best_route = current_route[:]
    best_dist = current_dist
    
    t = temp
    initial_temp = temp
    no_improve_count = 0
    
    for i in range(iterations):
        neighbor = move_func(current_route)
        neighbor_dist = tsp.route_length(neighbor)
        diff = neighbor_dist - current_dist
        
        if diff < 0 or (t > 0 and random.random() < math.exp(-diff / t)):
            current_route = neighbor
            current_dist = neighbor_dist
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = current_route[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # REHEATING: Podgrzej jeśli brak postępu
        if no_improve_count >= reheat_threshold:
            t = initial_temp * reheat_factor
            no_improve_count = 0
            # Wróć do najlepszego rozwiązania
            current_route = best_route[:]
            current_dist = best_dist
        else:
            t *= alpha
    
    return best_route, best_dist