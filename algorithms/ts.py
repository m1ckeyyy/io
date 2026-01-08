# -*- coding: utf-8 -*-
"""
Algorytm Przeszukiwania z Listą Tabu (TS - Tabu Search)
dla problemu komiwojażera (TSP).

Parametry:
- iterations: liczba iteracji
- tabu_size: długość listy tabu
- neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
- aspiration: czy używać kryterium aspiracji
- candidates_per_iter: liczba kandydatów sprawdzanych w każdej iteracji
"""
import random
from collections import deque
from utils.neighborhoods import NEIGHBORHOODS


def tabu_search(
    tsp,
    iterations=500,
    tabu_size=20,
    neighborhood="two_opt",
    aspiration=True,
    candidates_per_iter=20,
    no_improve_limit=None,
    use_nn_start=False
):
    """
    Przeszukiwanie z listą Tabu (TS)
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        iterations: maksymalna liczba iteracji
        tabu_size: długość listy tabu
        neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
        aspiration: czy używać kryterium aspiracji (akceptuj tabu jeśli lepsze od best)
        candidates_per_iter: liczba kandydatów do sprawdzenia w iteracji
        no_improve_limit: limit iteracji bez poprawy (None = brak)
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
    
    # Rozwiązanie startowe
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
    
    # Lista tabu - przechowuje krotki (tuple) reprezentujące trasę
    tabu_list = deque(maxlen=tabu_size)
    
    no_improve_count = 0
    
    # === PĘTLA GŁÓWNA TABU SEARCH ===
    for _ in range(iterations):
        best_candidate = None  # Najlepszy kandydat w tej iteracji
        best_candidate_dist = float('inf')
        best_candidate_tabu = False
        
        # --- KROK 1: Generuj i oceń sąsiadów (kandydatów) ---
        for _ in range(candidates_per_iter):
            # Wygeneruj losowego sąsiada
            candidate = move_func(current_route)
            candidate_tuple = tuple(candidate)  # Konwertuj na krotkę (hashable)
            is_tabu = candidate_tuple in tabu_list  # Czy na liście tabu?
            candidate_dist = tsp.route_length(candidate)  # Oceń kandydata
            
            # Sprawdź czy kandydat jest najlepszy wśród sprawdzonych
            if candidate_dist < best_candidate_dist:
                # === KRYTERIUM ASPIRACJI ===
                # Wyjątek: akceptuj ruch tabu jeśli daje NOWY NAJLEPSZY wynik
                if not is_tabu or (aspiration and candidate_dist < best_dist):
                    best_candidate = candidate
                    best_candidate_dist = candidate_dist
                    best_candidate_tabu = is_tabu
        
        # --- KROK 2: Wykonaj najlepszy znaleziony ruch ---
        if best_candidate is not None:
            current_route = best_candidate  # Przejdź do nowego rozwiązania
            current_dist = best_candidate_dist
            # Dodaj do listy tabu (zapobiega cofaniu się)
            tabu_list.append(tuple(current_route))
            
            # Aktualizuj najlepsze globalne rozwiązanie
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = current_route[:]
                no_improve_count = 0  # Reset licznika
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Sprawdź kryterium stopu
        if no_improve_limit and no_improve_count >= no_improve_limit:
            break
    
    return best_route, best_dist


def tabu_search_diversification(
    tsp,
    iterations=500,
    tabu_size=20,
    neighborhood="two_opt",
    diversification_threshold=50,
    diversification_strength=0.3
):
    """
    USPRAWNIENIE AUTORSKIE: TS z dywersyfikacją
    
    Gdy algorytm utknął (brak poprawy przez wiele iteracji),
    wykonujemy dywersyfikację - silne perturbacje rozwiązania
    aby eksplorować nowe regiony przestrzeni rozwiązań.
    
    Args:
        tsp: obiekt TSP
        iterations: liczba iteracji
        tabu_size: długość listy tabu
        neighborhood: typ sąsiedztwa
        diversification_threshold: iteracje bez poprawy do dywersyfikacji
        diversification_strength: siła perturbacji (0.0-1.0)
    
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
    
    tabu_list = deque(maxlen=tabu_size)
    no_improve_count = 0
    
    for _ in range(iterations):
        best_candidate = None
        best_candidate_dist = float('inf')
        
        for _ in range(20):
            candidate = move_func(current_route)
            if tuple(candidate) not in tabu_list:
                d = tsp.route_length(candidate)
                if d < best_candidate_dist:
                    best_candidate = candidate
                    best_candidate_dist = d
        
        if best_candidate:
            current_route = best_candidate
            current_dist = best_candidate_dist
            tabu_list.append(tuple(current_route))
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = current_route[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # DYWERSYFIKACJA: Silna perturbacja gdy utknęliśmy
        if no_improve_count >= diversification_threshold:
            # Liczba par do zamiany
            num_swaps = int(n * diversification_strength)
            for _ in range(max(1, num_swaps)):
                a, b = random.sample(range(n), 2)
                current_route[a], current_route[b] = current_route[b], current_route[a]
            
            current_dist = tsp.route_length(current_route)
            tabu_list.clear()  # Wyczyść listę tabu po dywersyfikacji
            no_improve_count = 0
    
    return best_route, best_dist