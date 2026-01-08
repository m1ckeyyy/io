# -*- coding: utf-8 -*-
"""
Algorytm Iteracyjnej Wspinaczki z Multistartem (IHC - Iterative Hill Climbing)
dla problemu komiwojażera (TSP).

Parametry:
- iterations: liczba iteracji dla każdego restartu
- restarts: liczba restartów (multistart)
- neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
- no_improve_limit: limit iteracji bez poprawy (opcjonalne kryterium stopu)
"""
import random
from utils.neighborhoods import NEIGHBORHOODS_DELTA, NEIGHBORHOODS


def iterative_hill_climbing(
    tsp,
    iterations=5000,
    restarts=20,
    neighborhood="two_opt",
    no_improve_limit=None,
    use_nn_start=False
):
    """
    Iteracyjna wspinaczka z multistartem (IHC)
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        iterations: maksymalna liczba iteracji na restart
        restarts: liczba restartów
        neighborhood: typ sąsiedztwa ("swap", "insert", "two_opt")
        no_improve_limit: limit iteracji bez poprawy (None = brak limitu)
        use_nn_start: czy używać rozwiązania NN jako startowego (USPRAWNIENIE 1)
    
    Returns:
        (best_route, best_length)
    """
    best_global_route = None
    best_global_length = float("inf")
    n = tsp.n
    
    # Wybór funkcji sąsiedztwa z delta evaluation
    if neighborhood in NEIGHBORHOODS_DELTA:
        neigh_delta_func = NEIGHBORHOODS_DELTA[neighborhood]
    else:
        neigh_delta_func = NEIGHBORHOODS_DELTA["two_opt"]
    
    # === PĘTLA GŁÓWNA: Wykonaj wiele restartów (multistart) ===
    for restart in range(restarts):
        
        # --- KROK 1: Generowanie trasy startowej ---
        if use_nn_start and restart == 0:
            # USPRAWNIENIE: Użyj rozwiązania NN jako punkt startowy
            from algorithms.nn import nearest_neighbor
            route, current_length = nearest_neighbor(tsp, start=random.randint(0, n-1))
            route = list(route)
        else:
            # Losowa permutacja miast jako trasa startowa
            route = list(range(n))
            random.shuffle(route)  # Losowe przemieszanie
            current_length = tsp.route_length(route)  # Oblicz długość
        
        no_improve_count = 0  # Licznik iteracji bez poprawy
        
        # --- KROK 2: Lokalna optymalizacja (hill climbing) ---
        for _ in range(iterations):
            # Generuj losowego sąsiada i oblicz zmianę kosztu (delta)
            new_route, delta = neigh_delta_func(route, tsp)
            
            # Akceptuj TYLKO jeśli sąsiad jest lepszy (delta < 0)
            # To jest kluczowa różnica od SA - brak akceptacji gorszych!
            if delta < 0:
                route = new_route  # Przyjmij nową trasę
                current_length += delta  # Zaktualizuj koszt
                no_improve_count = 0  # Reset licznika
            else:
                no_improve_count += 1  # Brak poprawy - zwiększ licznik
            
            # Kryterium stopu: zbyt długo bez poprawy
            if no_improve_limit and no_improve_count >= no_improve_limit:
                break  # Przerwij ten restart, zacznij następny
        
        # --- KROK 3: Aktualizacja najlepszego globalnego rozwiązania ---
        if current_length < best_global_length:
            best_global_route = route[:]  # Zapisz kopię trasy
            best_global_length = current_length
    
    return best_global_route, best_global_length


def ihc_with_intensification(
    tsp,
    iterations=5000,
    restarts=20,
    neighborhood="two_opt",
    intensification_threshold=0.01
):
    """
    USPRAWNIENIE AUTORSKIE: IHC z intensyfikacją
    
    Gdy znajdziemy dobre rozwiązanie, intensyfikujemy przeszukiwanie
    w jego okolicy używając wszystkich trzech typów sąsiedztwa.
    
    Args:
        tsp: obiekt TSP
        iterations: liczba iteracji
        restarts: liczba restartów
        neighborhood: główny typ sąsiedztwa
        intensification_threshold: próg poprawy do intensyfikacji (%)
    
    Returns:
        (best_route, best_length)
    """
    best_global_route = None
    best_global_length = float("inf")
    n = tsp.n
    
    neigh_delta_func = NEIGHBORHOODS_DELTA.get(neighborhood, NEIGHBORHOODS_DELTA["two_opt"])
    
    for restart in range(restarts):
        route = list(range(n))
        random.shuffle(route)
        current_length = tsp.route_length(route)
        
        for _ in range(iterations):
            new_route, delta = neigh_delta_func(route, tsp)
            
            if delta < 0:
                route = new_route
                current_length += delta
        
        # Sprawdź czy znaleźliśmy znaczącą poprawę
        if current_length < best_global_length:
            improvement = (best_global_length - current_length) / best_global_length if best_global_length < float("inf") else 1.0
            
            best_global_route = route[:]
            best_global_length = current_length
            
            # INTENSYFIKACJA: jeśli poprawa > próg, przeszukaj wszystkimi sąsiedztwami
            if improvement > intensification_threshold:
                for neigh_name in ["swap", "insert", "two_opt"]:
                    intensify_func = NEIGHBORHOODS_DELTA[neigh_name]
                    for _ in range(iterations // 3):
                        new_route, delta = intensify_func(route, tsp)
                        if delta < 0:
                            route = new_route
                            current_length += delta
                
                if current_length < best_global_length:
                    best_global_route = route[:]
                    best_global_length = current_length
    
    return best_global_route, best_global_length
