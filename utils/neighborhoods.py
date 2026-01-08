# -*- coding: utf-8 -*-
"""
Moduł zawierający trzy rodzaje ruchów (sąsiedztw) dla algorytmów TSP:
1. SWAP - zamiana dwóch miast miejscami
2. INSERT - wstawienie miasta w inne miejsce  
3. TWO-OPT - odwrócenie fragmentu trasy

Każda funkcja zwraca nową trasę (nie modyfikuje oryginalnej).
"""
import random


def swap(route):
    """
    Ruch SWAP: Zamienia dwa losowe miasta miejscami.
    Przykład: [1,2,3,4,5] -> [1,4,3,2,5] (zamiana pozycji 1 i 3)
    """
    n = len(route)
    if n < 2:
        return route[:]
    a, b = random.sample(range(n), 2)
    new = route[:]
    new[a], new[b] = new[b], new[a]
    return new


def insert(route):
    """
    Ruch INSERT: Wyjmuje miasto z jednej pozycji i wstawia w inną.
    Przykład: [1,2,3,4,5] -> [1,3,4,2,5] (miasto 2 przeniesione na pozycję 3)
    """
    n = len(route)
    if n < 2:
        return route[:]
    a, b = random.sample(range(n), 2)
    new = route[:]
    city = new.pop(a)
    new.insert(b, city)
    return new


def two_opt(route):
    """
    Ruch 2-OPT: Odwraca fragment trasy między dwoma punktami.
    Przykład: [1,2,3,4,5] -> [1,4,3,2,5] (odwrócenie fragmentu 2-3-4)
    """
    n = len(route)
    if n < 3:
        return route[:]
    a, b = sorted(random.sample(range(n), 2))
    if b - a < 2:  # za krótki fragment
        b = min(a + 2, n)
    new = route[:a] + route[a:b][::-1] + route[b:]
    return new


# ============ FUNKCJE Z DELTA EVALUATION (szybka ocena) ============

def swap_delta(route, tsp):
    """
    SWAP z szybką oceną przyrostową (delta evaluation).
    Zwraca (nowa_trasa, zmiana_kosztu).
    """
    n = len(route)
    if n < 2:
        return route[:], 0.0
    
    a, b = random.sample(range(n), 2)
    if a > b:
        a, b = b, a
    
    # Oblicz zmianę kosztu bez przeliczania całej trasy
    dm = tsp.dist_matrix
    
    # Sąsiedzi przed zamianą
    a_prev = route[(a - 1) % n]
    a_next = route[(a + 1) % n]
    b_prev = route[(b - 1) % n]
    b_next = route[(b + 1) % n]
    
    city_a = route[a]
    city_b = route[b]
    
    # Koszt przed zamianą
    if b == a + 1:  # sąsiednie miasta
        old_cost = dm[a_prev][city_a] + dm[city_a][city_b] + dm[city_b][b_next]
        new_cost = dm[a_prev][city_b] + dm[city_b][city_a] + dm[city_a][b_next]
    elif a == 0 and b == n - 1:  # pierwszy i ostatni
        old_cost = dm[route[-2]][city_b] + dm[city_b][city_a] + dm[city_a][route[1]]
        new_cost = dm[route[-2]][city_a] + dm[city_a][city_b] + dm[city_b][route[1]]
    else:
        old_cost = (dm[a_prev][city_a] + dm[city_a][a_next] + 
                    dm[b_prev][city_b] + dm[city_b][b_next])
        new_cost = (dm[a_prev][city_b] + dm[city_b][a_next] + 
                    dm[b_prev][city_a] + dm[city_a][b_next])
    
    delta = new_cost - old_cost
    
    new_route = route[:]
    new_route[a], new_route[b] = new_route[b], new_route[a]
    
    return new_route, delta


def insert_delta(route, tsp):
    """
    INSERT z szybką oceną przyrostową (delta evaluation).
    Zwraca (nowa_trasa, zmiana_kosztu).
    
    NAPRAWIONO: Obliczenia delty oparte na bezpośrednim porównaniu tras.
    """
    n = len(route)
    if n < 2:
        return route[:], 0.0
    
    # Generuj nową trasę
    new_route = insert(route)
    
    # Oblicz deltę jako różnicę kosztów
    old_cost = tsp.route_length(route)
    new_cost = tsp.route_length(new_route)
    delta = new_cost - old_cost
    
    return new_route, delta


def two_opt_delta(route, tsp):
    """
    2-OPT z szybką oceną przyrostową (delta evaluation).
    Zwraca (nowa_trasa, zmiana_kosztu).
    """
    n = len(route)
    if n < 3:
        return route[:], 0.0
    
    a, b = sorted(random.sample(range(n), 2))
    if b - a < 2:
        return route[:], 0.0
    
    dm = tsp.dist_matrix
    
    # Punkty brzegowe
    A = route[a - 1]
    B = route[a]
    C = route[b - 1]
    D = route[b % n]
    
    # Zmiana kosztu: usuwamy krawędzie (A-B) i (C-D), dodajemy (A-C) i (B-D)
    old_cost = dm[A][B] + dm[C][D]
    new_cost = dm[A][C] + dm[B][D]
    
    delta = new_cost - old_cost
    
    new_route = route[:]
    new_route[a:b] = reversed(new_route[a:b])
    
    return new_route, delta


# ============ SŁOWNIK SĄSIEDZTW ============

NEIGHBORHOODS = {
    "swap": swap,
    "insert": insert,
    "two_opt": two_opt
}

NEIGHBORHOODS_DELTA = {
    "swap": swap_delta,
    "insert": insert_delta,
    "two_opt": two_opt_delta
}
