# -*- coding: utf-8 -*-
"""
Algorytm Mrówkowy (ACO - Ant Colony Optimization)
dla problemu komiwojażera (TSP).

Dodatkowy (6.) algorytm wymagany przez projekt.

ACO symuluje zachowanie kolonii mrówek szukających najkrótszej drogi do pożywienia.
Mrówki zostawiają feromony na ścieżkach, które preferują kolejne mrówki.

Parametry:
- n_ants: liczba mrówek w kolonii
- n_iterations: liczba iteracji algorytmu
- alpha: wpływ feromonów na wybór ścieżki
- beta: wpływ heurystyki (odległości) na wybór ścieżki
- rho: współczynnik parowania feromonów (evaporation rate)
- q: stała do obliczania ilości deponowanych feromonów
"""
import random
import math


def ant_colony_optimization(
    tsp,
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    q=100.0,
    initial_pheromone=1.0,
    elitist_weight=0
):
    """
    Algorytm Mrówkowy (Ant Colony Optimization)
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        n_ants: liczba mrówek w kolonii
        n_iterations: liczba iteracji
        alpha: waga feromonów (im większa, tym większy wpływ feromonów)
        beta: waga heurystyki 1/odległość (im większa, tym bardziej zachłanny wybór)
        rho: współczynnik parowania (0-1, im większy, tym szybsze zapominanie)
        q: stała do obliczania deponowanych feromonów
        initial_pheromone: początkowa wartość feromonów na krawędziach
        elitist_weight: waga dla najlepszej mrówki (0 = brak elityzmu)
    
    Returns:
        (best_route, best_dist)
    """
    n = tsp.n
    dist_matrix = tsp.dist_matrix
    
    # === INICJALIZACJA ===
    
    # Macierz feromonów - początkowo równa wartość na wszystkich krawędziach
    # feromony[i][j] = ilość feromonu na drodze z miasta i do j
    pheromone = [[initial_pheromone for _ in range(n)] for _ in range(n)]
    
    # Macierz heurystyki - preferujemy krótsze krawędzie
    # heuristic[i][j] = 1/odległość (im bliżej, tym większa wartość)
    heuristic = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / dist_matrix[i][j]
    
    best_route = None
    best_dist = float('inf')
    
    # === PĘTLA GŁÓWNA ACO ===
    for iteration in range(n_iterations):
        all_routes = []  # Trasy wszystkich mrówek w tej iteracji
        all_distances = []  # Długości tras
        
        # --- KROK 1: Każda mrówka buduje swoją trasę ---
        for ant in range(n_ants):
            # Mrówka konstruuje trasę probabilistycznie
            route = _construct_solution(n, pheromone, heuristic, alpha, beta)
            dist = tsp.route_length(route)
            all_routes.append(route)
            all_distances.append(dist)
            
            # Aktualizuj najlepsze znalezione rozwiązanie
            if dist < best_dist:
                best_dist = dist
                best_route = route[:]
        
        # --- KROK 2: Parowanie feromonów (zapominanie) ---
        # Symuluje "wysychanie" feromonów - stare ścieżki tracą siłę
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)  # Redukuj o współczynnik rho
                pheromone[i][j] = max(pheromone[i][j], 0.0001)  # Min wartość
        
        # --- KROK 3: Depozycja feromonów przez mrówki ---
        # Im krótsza trasa, tym więcej feromonu zostawia mrówka
        for route, dist in zip(all_routes, all_distances):
            deposit = q / dist  # Więcej feromonu dla krótszych tras
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                # Dodaj feromon do krawędzi (w obu kierunkach)
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
        
        # --- KROK 4 (opcja): Elityzm - wzmocnij najlepszą trasę ---
        if elitist_weight > 0 and best_route:
            elite_deposit = elitist_weight * q / best_dist
            for i in range(n):
                a = best_route[i]
                b = best_route[(i + 1) % n]
                pheromone[a][b] += elite_deposit
                pheromone[b][a] += elite_deposit
    
    return best_route, best_dist


def _construct_solution(n, pheromone, heuristic, alpha, beta):
    """
    Konstruuje trasę dla pojedynczej mrówki używając reguły proporcjonalnej.
    
    Mrówka wybiera następne miasto probabilistycznie:
    P(i->j) = [feromon(i,j)]^alpha * [heurystyka(i,j)]^beta / suma
    
    alpha - waga feromonów (wpływ historii)
    beta - waga heurystyki (wpływ odległości)
    """
    # Losowy punkt startowy
    start = random.randint(0, n - 1)
    route = [start]
    visited = {start}
    current = start
    
    while len(route) < n:
        # Oblicz prawdopodobieństwa dla nieodwiedzonych miast
        probabilities = []
        unvisited = []
        
        for city in range(n):
            if city not in visited:
                # Reguła proporcjonalna: τ^α * η^β
                tau = pheromone[current][city] ** alpha
                eta = heuristic[current][city] ** beta
                prob = tau * eta
                probabilities.append(prob)
                unvisited.append(city)
        
        if not unvisited:
            break
        
        # Normalizacja i wybór
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            # Wybór ruletką
            next_city = random.choices(unvisited, weights=probabilities, k=1)[0]
        else:
            # Fallback: losowy wybór
            next_city = random.choice(unvisited)
        
        route.append(next_city)
        visited.add(next_city)
        current = next_city
    
    return route


def aco_with_local_search(
    tsp,
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    q=100.0,
    local_search_iters=50
):
    """
    ACO z lokalnym przeszukiwaniem (2-opt) po każdej konstrukcji.
    
    Ta wersja łączy ACO z lokalnym przeszukiwaniem, co często
    prowadzi do lepszych wyników niż samo ACO.
    """
    from utils.neighborhoods import two_opt
    
    n = tsp.n
    dist_matrix = tsp.dist_matrix
    
    pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
    heuristic = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / dist_matrix[i][j]
    
    best_route = None
    best_dist = float('inf')
    
    for iteration in range(n_iterations):
        all_routes = []
        all_distances = []
        
        for ant in range(n_ants):
            route = _construct_solution(n, pheromone, heuristic, alpha, beta)
            
            # Lokalne przeszukiwanie 2-opt
            current_dist = tsp.route_length(route)
            for _ in range(local_search_iters):
                new_route = two_opt(route)
                new_dist = tsp.route_length(new_route)
                if new_dist < current_dist:
                    route = new_route
                    current_dist = new_dist
            
            all_routes.append(route)
            all_distances.append(current_dist)
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = route[:]
        
        # Aktualizacja feromonów
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)
                pheromone[i][j] = max(pheromone[i][j], 0.0001)
        
        for route, dist in zip(all_routes, all_distances):
            deposit = q / dist
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
    
    return best_route, best_dist


def max_min_ant_system(
    tsp,
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    q=100.0
):
    """
    MAX-MIN Ant System (MMAS) - ulepszona wersja ACO.
    
    Różnice od podstawowego ACO:
    1. Tylko najlepsza mrówka deponuje feromony
    2. Wartości feromonów ograniczone do [tau_min, tau_max]
    3. Inicjalizacja z maksymalnym poziomem feromonów
    """
    n = tsp.n
    dist_matrix = tsp.dist_matrix
    
    # Szacunkowe tau_max i tau_min
    # Heurystyka NN daje przybliżenie długości optymalnej trasy
    from algorithms.nn import nearest_neighbor
    _, nn_dist = nearest_neighbor(tsp)
    
    tau_max = 1.0 / (rho * nn_dist) if nn_dist > 0 else 1.0
    tau_min = tau_max / (2 * n)
    
    # Inicjalizacja z tau_max
    pheromone = [[tau_max for _ in range(n)] for _ in range(n)]
    
    heuristic = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / dist_matrix[i][j]
    
    best_route = None
    best_dist = float('inf')
    iteration_best_route = None
    iteration_best_dist = float('inf')
    
    for iteration in range(n_iterations):
        iteration_best_route = None
        iteration_best_dist = float('inf')
        
        for ant in range(n_ants):
            route = _construct_solution(n, pheromone, heuristic, alpha, beta)
            dist = tsp.route_length(route)
            
            if dist < iteration_best_dist:
                iteration_best_dist = dist
                iteration_best_route = route[:]
            
            if dist < best_dist:
                best_dist = dist
                best_route = route[:]
        
        # Parowanie
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)
        
        # Tylko najlepsza mrówka (iteracji lub globalna) deponuje
        # Używamy globalnej najlepszej częściej w późniejszych iteracjach
        if random.random() < iteration / n_iterations:
            deposit_route = best_route
            deposit_dist = best_dist
        else:
            deposit_route = iteration_best_route
            deposit_dist = iteration_best_dist
        
        if deposit_route and deposit_dist > 0:
            deposit = q / deposit_dist
            for i in range(n):
                a = deposit_route[i]
                b = deposit_route[(i + 1) % n]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
        
        # Ograniczenie feromonów do [tau_min, tau_max]
        for i in range(n):
            for j in range(n):
                pheromone[i][j] = max(tau_min, min(tau_max, pheromone[i][j]))
    
    return best_route, best_dist
