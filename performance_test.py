import csv
import math
import os
import re
import random
import statistics
import time

from nearestNeighbor import initial_solution_nearest_neighbor, read_solomon_file as read_nn
from algo import generer_solution_initiale_randomisee
from read_file import lire_fichier_vrptw, calculer_matrice_distances
from tourGeant import initial_solution_hybrid_split, read_solomon_file as read_tour_geant
from solomon_inser import initial_solution_solomon_insertion, read_solomon_file as read_solomon


def euclidean_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def route_distance_clients(route, depot):
    total = 0.0
    current = depot
    for client in route:
        total += euclidean_distance(current, client)
        current = client
    total += euclidean_distance(current, depot)
    return total


def route_distance_ids(route, matrice_distances):
    total = 0.0
    for i in range(len(route) - 1):
        total += matrice_distances[route[i]][route[i + 1]]
    return total


def evaluate_nearest_neighbor(filename):
    depot, customers, vehicle_capacity = read_nn(filename)
    start = time.perf_counter()
    routes = initial_solution_nearest_neighbor(customers, depot, vehicle_capacity)
    elapsed = time.perf_counter() - start

    total_distance = sum(route_distance_clients(route, depot) for route in routes)
    return {
        'algorithm': 'NearestNeighbor',
        'routes': len(routes),
        'distance': total_distance,
        'time_s': elapsed,
    }


def evaluate_regret_k(filename, seed=42):
    random.seed(seed)
    coords, demands, windows, service_times, vehicle_capacity = lire_fichier_vrptw(filename)
    matrice_distances = calculer_matrice_distances(coords)
    depot = 0
    clients = list(range(1, len(coords)))

    start = time.perf_counter()
    routes = generer_solution_initiale_randomisee(clients, depot, matrice_distances, vehicle_capacity, demands, windows, service_times, K=3)
    elapsed = time.perf_counter() - start

    total_distance = sum(route_distance_ids(route, matrice_distances) for route in routes)
    return {
        'algorithm': 'RegretKRandomized',
        'routes': len(routes),
        'distance': total_distance,
        'time_s': elapsed,
    }


def evaluate_hybrid_split(filename, alpha=0.5, beta=0.5, seed=42):
    random.seed(seed)
    depot, customers, vehicle_capacity = read_tour_geant(filename)

    start = time.perf_counter()
    routes = initial_solution_hybrid_split(customers, depot, vehicle_capacity, alpha=alpha, beta=beta)
    elapsed = time.perf_counter() - start

    total_distance = sum(route_distance_clients(route, depot) for route in routes)
    return {
        'algorithm': 'GiantTourHybridSplit',
        'routes': len(routes),
        'distance': total_distance,
        'time_s': elapsed,
    }
    

def evaluate_solomon_insertion(filename):
    depot, customers, vehicle_capacity = read_solomon(filename)

    start = time.perf_counter()
    routes = initial_solution_solomon_insertion(customers, depot, vehicle_capacity)
    elapsed = time.perf_counter() - start

    total_distance = sum(route_distance_clients(route, depot) for route in routes)
    return {
        'algorithm': 'SolomonInsertion',
        'routes': len(routes),
        'distance': total_distance,
        'time_s': elapsed,
    }


def format_markdown_table(results):
    header = '| Fichier | Algorithme | Nombre de routes | Distance totale | Temps (s) |'
    separator = '|---|---|---|---|---|'
    rows = [header, separator]
    for result in results:
        rows.append(
            f"| {result['file']} | {result['algorithm']} | {result['routes']} | {result['distance']:.2f} | {result['time_s']:.4f} |"
        )
    return '\n'.join(rows)


def file_type(file_path):
    base = os.path.basename(file_path)
    m = re.match(r'^([A-Za-z]+)', base)
    return m.group(1) if m else 'UNKNOWN'


def summarize_by_file_type(results):
    agg = {}
    for r in results:
        t = file_type(r['file'])
        key = (t, r['algorithm'])
        agg.setdefault(key, []).append(r)

    summary = []
    for (t, algo), rows in sorted(agg.items()):
        summary.append({
            'file_type': t,
            'algorithm': algo,
            'count': len(rows),
            'avg_routes': statistics.mean(x['routes'] for x in rows),
            'avg_distance': statistics.mean(x['distance'] for x in rows),
            'avg_time_s': statistics.mean(x['time_s'] for x in rows),
        })
    return summary


def format_file_type_summary_table(summary):
    header = '| Type de fichier | Algorithme | Instances | Moyenne routes | Moyenne distance | Moyenne temps (s) |'
    separator = '|---|---|---|---|---|---|'
    rows = [header, separator]
    for line in summary:
        rows.append(
            f"| {line['file_type']} | {line['algorithm']} | {line['count']} | {line['avg_routes']:.2f} | {line['avg_distance']:.2f} | {line['avg_time_s']:.4f} |"
        )
    return '\n'.join(rows)


def main():
    archive_dir = 'Archive'
    fichiers = []
    for root, dirs, files in os.walk(archive_dir):
        if '__MACOSX' in root:
            continue
        for f in files:
            if not f.lower().endswith('.txt'):
                continue
            if f.startswith('._'):
                continue
            fichiers.append(os.path.join(root, f))
    fichiers.sort()

    if not fichiers:
        print(f"Aucun fichier .txt trouvé dans {archive_dir}")
        return

    results = []
    for filename in fichiers:
        print(f'Evaluation: {filename}')
        results.append({'file': filename, **evaluate_nearest_neighbor(filename)})
        results.append({'file': filename, **evaluate_regret_k(filename)})
        results.append({'file': filename, **evaluate_hybrid_split(filename)})
        results.append({'file': filename, **evaluate_solomon_insertion(filename)})
        print('  -> done')

    summary_md = format_markdown_table(results)
    type_summary = summarize_by_file_type(results)
    type_summary_md = format_file_type_summary_table(type_summary)

    print('\n=== Résultats au format Markdown ===\n')
    print(summary_md)
    print('\n=== Moyennes par type de fichier et par algorithme ===\n')
    print(type_summary_md)

    with open('performance_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Performance des algorithmes\n\n')
        f.write(summary_md)
        f.write('\n\n# Moyennes par type de fichier et par algorithme\n\n')
        f.write(type_summary_md)
        f.write('\n')

    print('\nRésumé sauvegardé dans performance_summary.md')


if __name__ == '__main__':
    main()
