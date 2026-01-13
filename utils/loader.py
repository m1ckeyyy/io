# -*- coding: utf-8 -*-
"""
Moduł do wczytywania instancji TSP z plików .tsp

Obsługuje format TSPLIB - standardowy format plików z danymi TSP.
"""


def load_tsp_file(path: str):
    """
    Wczytuje macierz odległości z pliku .tsp.
    
    Pliki w tym projekcie (Dane_TSP_*.tsp) zawierają pełną macierz odległości,
    gdzie separatorem dziesiętnym jest przecinek.
    Pierwszy wiersz i pierwsza kolumna to indeksy miast (do pominięcia).
    
    Args:
        path: ścieżka do pliku .tsp
    
    Returns:
        Macierz NxN (lista list floatów)
    """
    matrix = []
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Pomiń pierwszy wiersz (nagłówek z numerami kolumn)
        next(f, None)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Zamień polskie przecinki na kropki (format float)
            line = line.replace(",", ".")
            
            parts = line.split()
            
            # Wiersz powinien zaczynać się od indeksu, potem wartości
            # Ignorujemy pierwszy element (indeks wiersza)
            row_values = []
            # Startujemy od 1, bo parts[0] to numer miasta
            for p in parts[1:]:
                try:
                    val = float(p)
                    row_values.append(val)
                except ValueError:
                    pass
            
            if row_values:
                matrix.append(row_values)
    
    # Naprawa dla plików z brakującymi zerami na przekątnej (np. TSP_76)
    # Gdzie podwójny tabulator został zjedzony przez split()
    n = len(matrix)
    for i in range(n):
        if len(matrix[i]) == n - 1:
            # Brakuje jednego elementu: zera na przekątnej
            matrix[i].insert(i, 0.0)
            
    print(f"[DEBUG] Wczytano macierz o wymiarach: {len(matrix)} x {len(matrix[0]) if matrix else 0}")
    return matrix
