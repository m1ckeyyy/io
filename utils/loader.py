# -*- coding: utf-8 -*-
"""
Moduł do wczytywania instancji TSP z plików .tsp

Obsługuje format TSPLIB - standardowy format plików z danymi TSP.
"""


def load_tsp_file(path: str):
    """
    Wczytuje współrzędne miast z pliku TSP.
    
    Automatycznie wykrywa linie ze współrzędnymi (x, y).
    Obsługuje różne formaty: z przecinkami, spacjami, numerami linii.
    
    Args:
        path: ścieżka do pliku .tsp
    
    Returns:
        Lista krotek [(x1,y1), (x2,y2), ...] - współrzędne miast
    
    Przykład użycia:
        coords = load_tsp_file("instances/Dane_TSP_48.tsp")
        tsp = TSP(coords)
    """
    coords = []  # Lista współrzędnych do zwrócenia

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()  # Usuń białe znaki

            if not line:
                continue  # Pomiń puste linie

            # Zamień przecinki na spacje (różne formaty plików)
            line = line.replace(",", " ")
            parts = line.split()

            # Spróbuj wyłowić liczby z linii
            numbers = []
            for p in parts:
                try:
                    numbers.append(float(p))
                except ValueError:
                    pass  # Pomiń tekst (nagłówki, komentarze)

            # Jeśli są co najmniej 2 liczby → to współrzędne (x, y)
            # Bierzemy ostatnie dwie liczby (w formacie: nr x y)
            if len(numbers) >= 2:
                x = numbers[-2]  # Przedostatnia = x
                y = numbers[-1]  # Ostatnia = y
                coords.append((x, y))

    return coords
