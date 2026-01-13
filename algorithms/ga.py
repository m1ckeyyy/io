# -*- coding: utf-8 -*-
"""
Algorytm Genetyczny (GA - Genetic Algorithm)
dla problemu komiwojażera (TSP).

Zawiera:
- 3 metody selekcji rodziców: turniejowa, ruletkowa, rankingowa
- 3 metody krzyżowania: OX, PMX, CX
- 3 rodzaje mutacji: swap, insert, inversion (2-opt)

Parametry:
- pop_size: wielkość populacji
- generations: liczba generacji
- p_mut: prawdopodobieństwo mutacji
- p_cross: prawdopodobieństwo krzyżowania
- selection_type: "tournament", "roulette", "ranking"
- crossover_type: "ox", "pmx", "cx"
- mutation_type: "swap", "insert", "inversion"
- tournament_size: rozmiar turnieju (dla selekcji turniejowej)
- elitism: liczba najlepszych osobników przenoszonych bez zmian
"""
import random
from utils.neighborhoods import swap, insert, two_opt


def genetic_algorithm(
    tsp,
    pop_size=100,
    generations=200,
    p_mut=0.1,
    p_cross=0.9,
    selection_type="tournament",
    crossover_type="ox",
    mutation_type="swap",
    tournament_size=3,
    elitism=2,
    use_nn_start=False
):
    """
    Algorytm Genetyczny (GA)
    
    Args:
        tsp: obiekt TSP z macierzą odległości
        pop_size: wielkość populacji
        generations: liczba generacji
        p_mut: prawdopodobieństwo mutacji
        p_cross: prawdopodobieństwo krzyżowania
        selection_type: metoda selekcji ("tournament", "roulette", "ranking")
        crossover_type: metoda krzyżowania ("ox", "pmx", "cx")
        mutation_type: typ mutacji ("swap", "insert", "inversion")
        tournament_size: rozmiar turnieju
        elitism: liczba elitarnych osobników
        use_nn_start: czy zaszczepić populację rozwiązaniem NN (USPRAWNIENIE)
    
    Returns:
        (best_route, best_dist)
    """
    n = tsp.n
    
    # 1. Inicjalizacja populacji
    population = []
    
    # USPRAWNIENIE: Zaszczep populację rozwiązaniem NN
    if use_nn_start:
        from algorithms.nn import nearest_neighbor
        for start in range(min(5, n)):
            nn_route, _ = nearest_neighbor(tsp, start=start)
            population.append(list(nn_route))
    
    # Reszta populacji losowa
    while len(population) < pop_size:
        individual = list(range(n))
        random.shuffle(individual)
        population.append(individual)
    
    best_route = None
    best_dist = float('inf')
    
    # Słowniki funkcji
    selection_funcs = {
        "tournament": lambda pop, costs: tournament_selection(pop, costs, tournament_size),
        "roulette": roulette_selection,
        "ranking": ranking_selection
    }
    
    crossover_funcs = {
        "ox": order_crossover,
        "pmx": pmx_crossover,
        "cx": cycle_crossover
    }
    
    mutation_funcs = {
        "swap": swap,
        "insert": insert,
        "inversion": two_opt  # 2-opt jako mutacja inwersyjna
    }
    
    select_func = selection_funcs.get(selection_type, selection_funcs["tournament"])
    cross_func = crossover_funcs.get(crossover_type, crossover_funcs["ox"])
    mutate_func = mutation_funcs.get(mutation_type, mutation_funcs["swap"])
    
    # === PĘTLA GŁÓWNA: Ewolucja przez kolejne generacje ===
    for gen in range(generations):
        
        # --- KROK 1: Oceń wszystkich osobników (fitness = długość trasy) ---
        # Im krótsza trasa, tym lepszy fitness (minimalizujemy)
        costs = [tsp.route_length(ind) for ind in population]
        
        # Aktualizacja najlepszego globalnego wyniku
        for i, c in enumerate(costs):
            if c < best_dist:
                best_dist = c
                best_route = population[i][:]  # Zapisz kopię
        
        # --- KROK 2: Elityzm - zachowaj najlepszych bez zmian ---
        # Gwarantuje, że najlepsze rozwiązanie nie zostanie utracone
        elite_indices = sorted(range(len(costs)), key=lambda i: costs[i])[:elitism]
        elite = [population[i][:] for i in elite_indices]
        
        new_pop = elite[:]  # Nowa populacja zaczyna od elity
        
        # --- KROK 3: Twórz nowych potomków aż do wypełnienia populacji ---
        while len(new_pop) < pop_size:
            # SELEKCJA: Wybierz dwóch rodziców
            p1 = select_func(population, costs)  # Rodzic 1
            p2 = select_func(population, costs)  # Rodzic 2
            
            # KRZYŻOWANIE: Połącz geny rodziców
            if random.random() < p_cross:
                child = cross_func(p1, p2)  # Dziecko z genami obu rodziców
            else:
                child = p1[:]  # Brak krzyżowania - kopia rodzica
            
            # MUTACJA: Losowa modyfikacja potomka
            if random.random() < p_mut:
                child = mutate_func(child)
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    return best_route, best_dist


def ga_adaptive_mutation(
    tsp,
    pop_size=100,
    generations=200,
    initial_p_mut=0.1,
    selection_type="tournament",
    crossover_type="ox",
    use_nn_start=False
):
    """
    USPRAWNIENIE AUTORSKIE: GA z adaptacyjnym prawdopodobieństwem mutacji
    
    Prawdopodobieństwo mutacji rośnie gdy populacja staje się zbyt jednolita
    (brak różnorodności), a maleje gdy różnorodność jest wysoka.
    
    To pomaga:
    - Uniknąć przedwczesnej konwergencji
    - Zachować eksplorację gdy potrzeba
    - Intensyfikować gdy populacja jest różnorodna
    """
    n = tsp.n
    
    population = []
    if use_nn_start:
        from algorithms.nn import nearest_neighbor
        for start in range(min(5, n)):
            nn_route, _ = nearest_neighbor(tsp, start=start)
            population.append(list(nn_route))
            
    while len(population) < pop_size:
        individual = list(range(n))
        random.shuffle(individual)
        population.append(individual)
    
    best_route = None
    best_dist = float('inf')
    
    p_mut = initial_p_mut
    
    for gen in range(generations):
        costs = [tsp.route_length(ind) for ind in population]
        
        # Aktualizacja najlepszego
        for i, c in enumerate(costs):
            if c < best_dist:
                best_dist = c
                best_route = population[i][:]
        
        # ADAPTACJA: Oblicz różnorodność populacji
        diversity = _calculate_diversity(population)
        
        # Dostosuj prawdopodobieństwo mutacji
        if diversity < 0.3:  # Mała różnorodność
            p_mut = min(0.5, p_mut * 1.2)  # Zwiększ mutację
        elif diversity > 0.7:  # Duża różnorodność
            p_mut = max(0.01, p_mut * 0.8)  # Zmniejsz mutację
        
        # Elityzm
        elite_indices = sorted(range(len(costs)), key=lambda i: costs[i])[:2]
        elite = [population[i][:] for i in elite_indices]
        
        new_pop = elite[:]
        
        while len(new_pop) < pop_size:
            if selection_type == "tournament":
                p1 = tournament_selection(population, costs)
                p2 = tournament_selection(population, costs)
            elif selection_type == "roulette":
                p1 = roulette_selection(population, costs)
                p2 = roulette_selection(population, costs)
            else:
                p1 = ranking_selection(population, costs)
                p2 = ranking_selection(population, costs)
            
            if crossover_type == "ox":
                child = order_crossover(p1, p2)
            elif crossover_type == "pmx":
                child = pmx_crossover(p1, p2)
            else:
                child = cycle_crossover(p1, p2)
            
            if random.random() < p_mut:
                child = swap(child)
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    return best_route, best_dist


def _calculate_diversity(population):
    """
    Oblicza różnorodność populacji jako procent unikalnych krawędzi.
    """
    if not population:
        return 0.0
    
    n = len(population[0])
    all_edges = set()
    
    for ind in population:
        for i in range(n):
            edge = (min(ind[i], ind[(i+1) % n]), max(ind[i], ind[(i+1) % n]))
            all_edges.add(edge)
    
    # Maksymalna możliwa liczba krawędzi
    max_edges = n * (n - 1) // 2
    
    return len(all_edges) / max_edges if max_edges > 0 else 0.0


# --- METODY SELEKCJI ---

def tournament_selection(pop, costs, k=3):
    """Selekcja turniejowa: wybiera najlepszego z k losowych osobników."""
    idx = random.sample(range(len(pop)), min(k, len(pop)))
    winner = min(idx, key=lambda i: costs[i])
    return pop[winner][:]


def roulette_selection(pop, costs):
    """Selekcja ruletkowa: prawdopodobieństwo proporcjonalne do fitness."""
    max_c = max(costs)
    fitness = [max_c - c + 1 for c in costs]  # +1 aby uniknąć zerowych wartości
    total = sum(fitness)
    probs = [f / total for f in fitness]
    chosen = random.choices(pop, weights=probs, k=1)[0]
    return chosen[:]


def ranking_selection(pop, costs):
    """Selekcja rankingowa: prawdopodobieństwo proporcjonalne do rangi."""
    n = len(pop)
    # Sortuj indeksy od najgorszego (najwyższy koszt) do najlepszego
    sorted_indices = sorted(range(n), key=lambda k: costs[k], reverse=True)
    # Rangi: najgorszy ma rangę 1, najlepszy ma rangę n
    ranks = list(range(1, n + 1))
    total_ranks = sum(ranks)
    probs = [r / total_ranks for r in ranks]
    chosen_idx = random.choices(sorted_indices, weights=probs, k=1)[0]
    return pop[chosen_idx][:]


# --- METODY KRZYŻOWANIA ---

def order_crossover(p1, p2):
    """
    Order Crossover (OX): Kopiuje segment z P1, resztę uzupełnia z P2.
    """
    size = len(p1)
    if size < 2:
        return p1[:]
    
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    
    # Elementy z P2 które nie są w skopiowanym segmencie
    p2_remaining = [x for x in p2 if x not in child[a:b]]
    
    cursor = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_remaining[cursor]
            cursor += 1
    
    return child


def pmx_crossover(p1, p2):
    """
    Partially Mapped Crossover (PMX): Mapowanie pozycji między rodzicami.
    """
    size = len(p1)
    if size < 2:
        return p1[:]
    
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    
    # Mapowanie
    for i in range(a, b):
        if p2[i] not in child[a:b]:
            curr = p1[i]
            idx = p2.index(curr)
            while a <= idx < b:
                curr = p1[idx]
                idx = p2.index(curr)
            child[idx] = p2[i]
    
    # Uzupełnij pozostałe
    for i in range(size):
        if child[i] is None:
            child[i] = p2[i]
    
    return child


def cycle_crossover(p1, p2):
    """
    Cycle Crossover (CX): Kopiuje cykle naprzemiennie z P1 i P2.
    """
    size = len(p1)
    child = [None] * size
    cycle = 0
    
    while None in child:
        idx = child.index(None)
        start_val = p1[idx]
        
        # Śledź cykl
        while child[idx] is None:
            if cycle % 2 == 0:
                child[idx] = p1[idx]
            else:
                child[idx] = p2[idx]
            
            # Znajdź pozycję wartości z P2 w P1
            val_from_p2 = p2[idx]
            if val_from_p2 in p1:
                idx = p1.index(val_from_p2)
            else:
                break
        
        cycle += 1
    
    return child