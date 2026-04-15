"""
Test Script: Test any initial solution algorithm with Simulated Annealing

You can run this script in two ways:
1. Interactive mode (default):
   python test_RS.py
   
2. Command-line mode:
   python test_RS.py <filename> <algorithm>

Supported algorithms:
  - NN or nearest_neighbor  : Nearest Neighbor
  - S or solomon_insertion  : Solomon Insertion
  - TG or tour_geant       : Tour Géant (Hybrid Split)
  - R or regret            : Regret-K Heuristic
  - CW or clarke_wright    : Clarke-Wright Savings Algorithm

Examples:
  python test_RS.py ./Archive/C101.txt NN
  python test_RS.py ./Archive/C102.txt S
  python test_RS.py ./Archive/R101.txt TG
  python test_RS.py ./Archive/RC101.txt R
  python test_RS.py ./Archive/C201.txt CW
"""

import math
import sys
import time
from typing import List, Tuple, Callable

# Import different algorithms
from nearestNeighbor import initial_solution_nearest_neighbor as nn_algorithm
from solomon_inser import initial_solution_solomon_insertion as solomon_algorithm
from tourGeant import initial_solution_hybrid_split as tour_geant_algorithm
from regret_algorithm.algo import generer_solution_initiale_randomisee
from regret_algorithm.clarke_wright import generer_solution_clarke_wright

# Import SA and structures
from RS_final import (
    simulated_annealing,
    SAConfig,
    Customer,
    Instance,
    is_feasible,
)


class Client:
    """Compatibility wrapper for different implementations"""
    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


def distance(c1, c2):
    """Calculate Euclidean distance between two clients"""
    return math.hypot(c1.x - c2.x, c1.y - c2.y)


def read_solomon_file(filename):
    """
    Read a Solomon VRPTW file and return depot, customers, and vehicle capacity
    """
    clients = []
    vehicle_capacity = None

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']

        # Find vehicle capacity
        for i, line in enumerate(lines):
            if line.upper().startswith("VEHICLE"):
                cap_line = lines[i+2]
                vehicle_capacity = int(cap_line.split()[1])
                break

        # Find customer data start
        cust_start = None
        for i, line in enumerate(lines):
            if line.upper().startswith("CUSTOMER"):
                cust_start = i + 2
                break

        if cust_start is None:
            raise ValueError("Customer data not found in file!")

        # Read customer data
        for line in lines[cust_start:]:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_time = float(parts[5])
                service_time = float(parts[6])
                clients.append(Client(id, x, y, demand, ready_time, due_time, service_time))

    depot = clients[0]
    customers = clients[1:]

    return depot, customers, vehicle_capacity


def convert_to_sa_format(routes: List[List[Client]], depot: Client) -> List[List[int]]:
    """
    Convert routes with Client objects to SA format with integer indices
    Format: [[0, c1_id, c2_id, ..., 0], ...]
    """
    sa_routes = []
    for route in routes:
        sa_route = [0] + [c.id for c in route] + [0]
        sa_routes.append(sa_route)
    return sa_routes


def convert_to_instance(depot: Client, customers: list, vehicle_capacity: float) -> Instance:
    """Convert our Client objects to SA Instance format"""
    sa_depot = Customer(
        id=depot.id,
        x=depot.x,
        y=depot.y,
        demand=depot.demand,
        ready_time=depot.ready_time,
        due_date=depot.due_time,
        service_time=depot.service_time
    )

    sa_customers = [
        Customer(
            id=c.id,
            x=c.x,
            y=c.y,
            demand=c.demand,
            ready_time=c.ready_time,
            due_date=c.due_time,
            service_time=c.service_time
        )
        for c in customers
    ]

    return Instance(
        depot=sa_depot,
        customers=sa_customers,
        vehicle_capacity=vehicle_capacity,
        num_vehicles=len(customers)  # Upper bound
    )


def regret_algorithm_wrapper(customers: list, depot: Client, vehicle_capacity: float) -> List[List[Client]]:
    """
    Wrapper function to adapt the Regret-K algorithm to work with our unified interface.
    
    Converts Client objects to the format expected by the regret algorithm,
    runs it, and converts the result back.
    
    Args:
        customers: List of Client objects
        depot: Depot Client object
        vehicle_capacity: Maximum vehicle capacity
        
    Returns:
        List of routes, where each route is a list of Client objects
    """
    # Build distance matrix (n+1 x n+1, where 0 is depot)
    n = len(customers)
    matrice_distances = [[0.0] * (n + 1) for _ in range(n + 1)]
    
    # Compute distances between all pairs
    all_nodes = [depot] + customers
    for i in range(n + 1):
        for j in range(n + 1):
            matrice_distances[i][j] = distance(all_nodes[i], all_nodes[j])
    
    # Build demand, time window, and service time dictionaries
    demandes = {}
    fenetres_temps = {}
    temps_service = {}
    
    demandes[0] = 0  # Depot has 0 demand
    fenetres_temps[0] = (depot.ready_time, depot.due_time)
    temps_service[0] = depot.service_time
    
    for i, client in enumerate(customers):
        client_id = i + 1  # Client IDs start from 1
        demandes[client_id] = client.demand
        fenetres_temps[client_id] = (client.ready_time, client.due_time)
        temps_service[client_id] = client.service_time
    
    # List of client IDs (1 to n)
    tous_les_clients = list(range(1, n + 1))
    
    # Call the regret algorithm
    routes_ids = generer_solution_initiale_randomisee(
        tous_les_clients=tous_les_clients,
        depot=0,
        matrice_distances=matrice_distances,
        capacite_max=vehicle_capacity,
        demandes=demandes,
        fenetres_temps=fenetres_temps,
        temps_service=temps_service,
        K=3
    )
    
    # Convert routes from ID format back to Client objects
    # routes_ids: [[0, 5, 2, 0], [0, 1, 3, 4, 0], ...]
    # We need: [[client5, client2], [client1, client3, client4], ...]
    routes = []
    for route_ids in routes_ids:
        # Skip the depot markers (0)
        route_clients = []
        for client_id in route_ids[1:-1]:  # Exclude first and last depot
            route_clients.append(customers[client_id - 1])  # Convert ID to index
        if route_clients:  # Only add non-empty routes
            routes.append(route_clients)
    
    return routes


def clarke_wright_wrapper(customers: list, depot: Client, vehicle_capacity: float) -> List[List[Client]]:
    """
    Wrapper function to adapt the Clarke-Wright algorithm to work with our unified interface.
    
    Converts Client objects to the format expected by the algorithm,
    runs it, and converts the result back.
    
    Args:
        customers: List of Client objects
        depot: Depot Client object
        vehicle_capacity: Maximum vehicle capacity
        
    Returns:
        List of routes, where each route is a list of Client objects
    """
    # Build distance matrix (n+1 x n+1, where 0 is depot)
    n = len(customers)
    matrice_distances = [[0.0] * (n + 1) for _ in range(n + 1)]
    
    # Compute distances between all pairs
    all_nodes = [depot] + customers
    for i in range(n + 1):
        for j in range(n + 1):
            matrice_distances[i][j] = distance(all_nodes[i], all_nodes[j])
    
    # Build demand, time window, and service time dictionaries
    demandes = {}
    fenetres_temps = {}
    temps_service = {}
    
    demandes[0] = 0  # Depot has 0 demand
    fenetres_temps[0] = (depot.ready_time, depot.due_time)
    temps_service[0] = depot.service_time
    
    for i, client in enumerate(customers):
        client_id = i + 1  # Client IDs start from 1
        demandes[client_id] = client.demand
        fenetres_temps[client_id] = (client.ready_time, client.due_time)
        temps_service[client_id] = client.service_time
    
    # List of client IDs (1 to n)
    tous_les_clients = list(range(1, n + 1))
    
    # Call the Clarke-Wright algorithm
    routes_ids = generer_solution_clarke_wright(
        tous_les_clients=tous_les_clients,
        depot=0,
        matrice_distances=matrice_distances,
        capacite_max=vehicle_capacity,
        demandes=demandes,
        fenetres_temps=fenetres_temps,
        temps_service=temps_service
    )
    
    # Convert routes from ID format back to Client objects
    # routes_ids: [[0, 5, 2, 0], [0, 1, 3, 4, 0], ...]
    # We need: [[client5, client2], [client1, client3, client4], ...]
    routes = []
    for route_ids in routes_ids:
        # Skip the depot markers (0)
        route_clients = []
        for client_id in route_ids[1:-1]:  # Exclude first and last depot
            route_clients.append(customers[client_id - 1])  # Convert ID to index
        if route_clients:  # Only add non-empty routes
            routes.append(route_clients)
    
    return routes


def run_algorithm(algorithm_name: str, customers: list, depot: Client, vehicle_capacity: float) -> List[List[Client]]:
    """
    Run the specified initial solution algorithm
    
    Returns:
        List of routes, each route is a list of Client objects
    """
    if algorithm_name.lower() == 'nearest_neighbor':
        print(f"🔄 Running Nearest Neighbor algorithm...")
        return nn_algorithm(customers, depot, vehicle_capacity)

    elif algorithm_name.lower() == 'solomon_insertion':
        print(f"🔄 Running Solomon Insertion algorithm...")
        return solomon_algorithm(customers, depot, vehicle_capacity)

    elif algorithm_name.lower() == 'tour_geant':
        print(f"🔄 Running Tour Géant (Hybrid Split) algorithm...")
        return tour_geant_algorithm(customers, depot, vehicle_capacity)

    elif algorithm_name.lower() == 'regret':
        print(f"🔄 Running Regret-K algorithm...")
        return regret_algorithm_wrapper(customers, depot, vehicle_capacity)

    elif algorithm_name.lower() == 'clarke_wright':
        print(f"🔄 Running Clarke-Wright Savings algorithm...")
        return clarke_wright_wrapper(customers, depot, vehicle_capacity)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def print_routes(routes: List[List[int]], sa_routes: List[List[int]], operator_stats: dict, inst: Instance):
    """Print detailed route information"""
    print("\n" + "="*70)
    print("ROUTES (with depot at start and end):")
    print("="*70)

    total_distance = 0
    total_demand = 0

    for i, route in enumerate(sa_routes):
        # Calculate load and distance
        load = 0
        route_distance = 0

        # Exclude depot (0) from load calculation
        for j in range(1, len(route) - 1):
            load += inst.node(route[j]).demand

        # Calculate distance including return to depot
        for j in range(len(route) - 1):
            route_distance += inst.distance(route[j], route[j + 1])

        total_distance += route_distance
        total_demand += load

        print(f"Route {i+1:2d} | Load: {load:6.1f} | Distance: {route_distance:8.2f} | Path: {route}")

    print("="*70)
    print(f"Total Routes: {len(sa_routes)}")
    print(f"Total Distance: {total_distance:.2f}")
    print(f"Swapping Stats: {operator_stats}")
    print(f"Total Demand: {total_demand:.1f}")
    print("="*70)


def main():
    # Get inputs from command-line or interactive mode
    if len(sys.argv) >= 3:
        # Command-line mode
        filename = sys.argv[1]
        algorithm_input = sys.argv[2]
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("VRPTW INITIAL SOLUTION + SIMULATED ANNEALING TEST")
        print("="*70)
        
        # Get filename
        while True:
            filename = "./Archive/"+input("\n📁 Enter the Solomon file (C101): ").strip()+".txt"
            if filename:
                break
            print("   ⚠️  Please enter a valid file path")
        
        # Get algorithm with shorthand support
        print("\nAvailable algorithms:")
        print("  NN : Nearest Neighbor")
        print("  S  : Solomon Insertion")
        print("  TG : Tour Géant (Hybrid Split)")
        print("  R  : Regret-K Heuristic")
        print("  CW : Clarke-Wright Savings Algorithm")
        
        while True:
            algorithm_input = input("\n🔧 Enter algorithm (NN/S/TG/R/CW): ").strip().upper()
            if algorithm_input in ['NN', 'S', 'TG', 'R', 'CW']:
                break
            print("   ⚠️  Invalid algorithm. Please choose from: NN, S, TG, R, CW")
    
    # Map shorthand to full algorithm names
    algorithm_map = {
        'NN': 'nearest_neighbor',
        'S': 'solomon_insertion',
        'TG': 'tour_geant',
        'R': 'regret',
        'CW': 'clarke_wright',
        'NEAREST_NEIGHBOR': 'nearest_neighbor',
        'SOLOMON_INSERTION': 'solomon_insertion',
        'TOUR_GEANT': 'tour_geant',
        'REGRET': 'regret',
        'CLARKE_WRIGHT': 'clarke_wright'
    }
    
    algorithm = algorithm_map.get(algorithm_input.upper(), algorithm_input.lower())

    try:
        # Read instance
        print(f"\n📂 Reading file: {filename}")
        depot, customers, vehicle_capacity = read_solomon_file(filename)
        print(f"   ✓ Loaded {len(customers)} customers, Capacity: {vehicle_capacity}")

        # Run initial solution algorithm
        print(f"\n{'='*70}")
        routes = run_algorithm(algorithm, customers, depot, vehicle_capacity)
        print(f"   ✓ Generated {len(routes)} routes")

        # Display initial solution
        print("\n" + "="*70)
        print("INITIAL SOLUTION:")
        print("="*70)
        sa_routes = convert_to_sa_format(routes, depot)
        inst = convert_to_instance(depot, customers, vehicle_capacity)

        total_dist = 0
        for i, route in enumerate(sa_routes):
            load = sum(inst.node(route[j]).demand for j in range(1, len(route) - 1))
            route_dist = sum(inst.distance(route[j], route[j+1]) for j in range(len(route) - 1))
            total_dist += route_dist
            print(f"Route {i+1:2d} | Load: {load:6.1f} | Distance: {route_dist:8.2f} | Path: {route}")

        print("-"*70)
        print(f"Initial Total Distance: {total_dist:.2f} | Routes: {len(sa_routes)}")
        print(f"Initial Feasible: {'✓' if is_feasible(sa_routes, inst) else '✗'}")

        # Run Simulated Annealing
        print(f"\n{'='*70}")
        print("RUNNING SIMULATED ANNEALING...")
        print("="*70)

        config = SAConfig(
            T_init=100.0,
            T_min=0.01,
            alpha=0.995,
            max_iter=100_000,
            penalty_weight=1000.0,
            target_acceptance=0.20,
            adapt_interval=500,
            reheat_patience=3000,
            reheat_factor=2.0,
            reaction_factor=0.15,
            segment_update=200,
            verbose=True,
            log_interval=5000,
        )

        result = simulated_annealing(sa_routes, inst, config)

        # Display final solution
        print(f"\n{'='*70}")
        print("FINAL SOLUTION (after Simulated Annealing):")
        print("="*70)
        print_routes(routes, result.best_solution, result.operator_stats, inst)

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print("="*70)
        print(f"Algorithm: {algorithm}")
        print(f"File: {filename}")
        print(f"Customers: {len(customers)}")
        print(f"Initial Cost: {total_dist:.2f}")
        print(f"Final Cost: {result.best_cost:.2f}")
        print(f"Improvement: {((total_dist - result.best_cost) / total_dist * 100):.2f}% {'↓' if result.best_cost < total_dist else '↑'}")
        print(f"Feasible: {'✓' if result.feasible else '✗'}")
        print(f"Final Routes: {len(result.best_solution)}")
        print(f"Elapsed Time: {result.elapsed:.2f}s")
        print("="*70)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
