# -*- coding: utf-8 -*-
"""
Klasa TSP - reprezentacja problemu komiwojażera.

Przechowuje współrzędne miast i oblicza macierz odległości.
Służy jako kontener danych dla wszystkich algorytmów.
"""
import math


class TSP:
    """
    Klasa reprezentująca instancję problemu komiwojażera (TSP).
    
    Attributes:
        coords: lista współrzędnych miast [(x1,y1), (x2,y2), ...]
        n: liczba miast
        dist_matrix: macierz odległości n x n
    """
    
    def __init__(self, data):
        """
        Inicjalizacja instancji TSP.
        
        Args:
            data: macierz odległości NxN LUB lista współrzędnych
        """
        # Sprawdzamy czy to macierz (lista list) czy lista krotek
        if data and isinstance(data[0], list):
            self.dist_matrix = data
            self.n = len(data)
            self.coords = None  # Brak współrzędnych, mamy gotową macierz
        else:
            self.coords = data
            self.n = len(data)
            self.dist_matrix = self._compute_dist_matrix()

    def _compute_dist_matrix(self):
        """
        Oblicza macierz odległości euklidesowych między wszystkimi miastami.
        
        Returns:
            Macierz n x n gdzie mat[i][j] = odległość między miastem i oraz j
        """
        n = self.n
        mat = [[0.0] * n for _ in range(n)]  # Tworzy pustą macierz n x n
        
        # Oblicz odległość dla każdej pary miast
        for i in range(n):
            for j in range(n):
                x1, y1 = self.coords[i]  # Współrzędne miasta i
                x2, y2 = self.coords[j]  # Współrzędne miasta j
                # Odległość euklidesowa: sqrt((x2-x1)² + (y2-y1)²)
                mat[i][j] = math.hypot(x1 - x2, y1 - y2)
        return mat

    def route_length(self, route):
        """
        Oblicza całkowitą długość trasy (cyklu).
        
        Args:
            route: lista indeksów miast w kolejności odwiedzania
        
        Returns:
            Suma odległości wszystkich krawędzi w trasie (włącznie z powrotem)
        """
        total = 0.0
        # Dodaj odległości między kolejnymi miastami
        for i in range(len(route)):
            a = route[i]  # Aktualne miasto
            b = route[(i + 1) % len(route)]  # Następne miasto (% zapewnia powrót)
            total += self.dist_matrix[a][b]
        return total
