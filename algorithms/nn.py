# -*- coding: utf-8 -*-
"""
Algorytm Nearest Neighbor (NN) - Najbliższego Sąsiada
dla problemu komiwojażera (TSP).

Algorytm zachłanny - w każdym kroku wybiera najbliższe nieodwiedzone miasto.
Złożoność: O(n²) gdzie n = liczba miast

Parametry:
- start: miasto startowe (domyślnie 0)
"""


def nearest_neighbor(tsp, start=0):
    """
    Algorytm najbliższego sąsiada (NN).
    
    Buduje trasę wybierając zawsze najbliższe nieodwiedzone miasto.
    Prosty, szybki, ale nie gwarantuje optimum.
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        start: indeks miasta startowego (0 do n-1)
    
    Returns:
        (route, total_length): znaleziona trasa i jej długość
    """
    n = tsp.n  # Liczba miast
    visited = [False] * n  # Tablica odwiedzonych miast
    route = [start]  # Trasa startuje z wybranego miasta
    visited[start] = True  # Oznacz miasto startowe jako odwiedzone

    total_length = 0.0  # Całkowita długość trasy
    current = start  # Aktualne miasto (pozycja komiwojażera)

    # Odwiedź wszystkie pozostałe miasta (n-1 razy)
    for _ in range(n - 1):
        best_city = -1  # Najlepsze (najbliższe) miasto
        best_dist = float("inf")  # Odległość do najlepszego miasta
        row = tsp.dist_matrix[current]  # Wiersz odległości z aktualnego miasta

        # Przeszukaj wszystkie miasta i znajdź najbliższe nieodwiedzone
        for city in range(n):
            if not visited[city] and row[city] < best_dist:
                best_dist = row[city]
                best_city = city

        # Przenieś się do najbliższego miasta
        visited[best_city] = True  # Oznacz jako odwiedzone
        route.append(best_city)  # Dodaj do trasy
        total_length += best_dist  # Dodaj odległość
        current = best_city  # Zmień pozycję

    # Wróć do miasta startowego (zamknij cykl)
    total_length += tsp.dist_matrix[current][start]
    return route, total_length
