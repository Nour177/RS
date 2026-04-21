import math
import random

# Client class
class Client:
    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


# Distance function
def distance(c1, c2):
    return math.hypot(c1.x - c2.x, c1.y - c2.y)

def simulate_route(route, depot, vehicle_capacity):
    """
    Simule une route de A à Z.
    Retourne (True, distance_totale) si la route respecte la capacité et le temps.
    Retourne (False, inf) si la route est illégale.
    """
    load = sum(c.demand for c in route)
    if load > vehicle_capacity:
        return False, float('inf')

    current_time = 0
    current_location = depot
    route_dist = 0

    for client in route:
        travel_time = distance(current_location, client)
        arrival_time = current_time + travel_time

        # Si on arrive après la fermeture de la fenêtre, la route est invalide
        if arrival_time > client.due_time:
            return False, float('inf')

        # Mise à jour de l'horloge et de la distance
        current_time = max(arrival_time, client.ready_time) + client.service_time
        route_dist += travel_time
        current_location = client

    # Retour au dépôt à la fin de la journée
    route_dist += distance(current_location, depot)
    return True, route_dist


def initial_solution_solomon_insertion(clients, depot, vehicle_capacity):
    """
    Génère une solution initiale avec l'algorithme d'insertion de Solomon.
    """
    unvisited = clients[:]
    routes = []

    while unvisited:
        route = []
        
        # 1. La Graine : On amorce le camion avec le client le plus urgent
        # (Celui dont la fenêtre de temps se ferme le plus tôt)
        seed_client = min(unvisited, key=lambda c: c.due_time)
        route.append(seed_client)
        unvisited.remove(seed_client)

        while True:
            best_client = None
            best_pos = -1
            best_cost = float('inf')
            
            # Calcul de la distance de la route AVANT insertion
            _, base_dist = simulate_route(route, depot, vehicle_capacity)

            # 2. Chercher la meilleure insertion possible
            for client in unvisited:
                # On teste l'insertion à toutes les positions possibles (0, 1, 2...)
                for pos in range(len(route) + 1):
                    # Création d'une copie virtuelle de la route avec le client inséré
                    temp_route = route[:pos] + [client] + route[pos:]
                    
                    # 3. L'insertion est-elle légale ?
                    is_valid, new_dist = simulate_route(temp_route, depot, vehicle_capacity)
                    
                    if is_valid:
                        # Le "coût" de l'insertion est le détour géométrique provoqué
                        insertion_cost = new_dist - base_dist 

                        if insertion_cost < best_cost:
                            best_cost = insertion_cost
                            best_client = client
                            best_pos = pos

            # 4. Appliquer la meilleure insertion trouvée
            if best_client is not None:
                route.insert(best_pos, best_client)
                unvisited.remove(best_client)
            else:
                # Si plus aucun client ne peut rentrer (bloqué par temps ou capacité)
                # On ferme ce camion et on sort de la boucle interne
                break
        
        routes.append(route)

    return routes
# Read VRPTW file
def read_solomon_file(filename):
    clients = []
    vehicle_capacity = None

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']

        #Find vehicle capacity
        for i, line in enumerate(lines):
            if line.upper().startswith("VEHICLE"):
                cap_line = lines[i+2] 
                vehicle_capacity = int(cap_line.split()[1])
                break

        #Find customer data start
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




# Main program
if __name__ == "__main__":

    filename = "./Archive/C101.txt"
    depot, customers, vehicle_capacity = read_solomon_file(filename)
    routes = initial_solution_solomon_insertion(customers, depot, vehicle_capacity)
    print("Number of routes:", len(routes))

    total_distance = 0

    for i, route in enumerate(routes):
        load = sum(c.demand for c in route)
        route_with_depot = [0] + [c.id for c in route] + [0]
        print(f"Route {i+1} | Load {load} :", route_with_depot)

        current_location = depot
        route_distance = 0
        for c in route:
            route_distance += distance(current_location, c)
            current_location = c
        route_distance += distance(current_location, depot)
        print(f"  Distance: {route_distance:.2f}")
        total_distance += route_distance

    print("Total distance:", total_distance)

    