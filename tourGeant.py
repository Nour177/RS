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

# --- TOUR GÉANT + SPLIT ---


def get_normalized_angle(client, depot):
    """Calcule l'angle polaire et le ramène sur une échelle de 0 à 1."""
    angle = math.atan2(client.y - depot.y, client.x - depot.x)
    if angle < 0:
        angle += 2 * math.pi # Convertir l'angle négatif en positif (0 à 2*pi)
    return angle / (2 * math.pi)

def initial_solution_hybrid_split(clients, depot, vehicle_capacity, alpha=0.5, beta=0.5):
    """
    Génère une solution initiale via 'Giant Tour + Split' avec un tri Hybride.
    alpha: Poids donné à la position spatiale (Angle)
    beta: Poids donné à la chronologie (Ready Time)
    """
    if not clients:
        return []

    # 1. Trouver le max_ready_time pour la normalisation
    max_ready_time = max(c.ready_time for c in clients)
    if max_ready_time == 0:
        max_ready_time = 1 # Éviter la division par zéro

    # 2. Fonction de calcul du score hybride
    def hybrid_score(client):
        norm_angle = get_normalized_angle(client, depot)
        norm_time = client.ready_time / max_ready_time
        noise = random.uniform(-0.05, 0.05)
        return (alpha * norm_angle) + (beta * norm_time) + noise
        # 3. Créer le Tour Géant en triant par ce score hybride
    giant_tour = sorted(clients, key=hybrid_score)
    
    n = len(giant_tour)
    # V[i] stocke le coût minimum pour router les 'i' premiers clients
    V = [float('inf')] * (n + 1)
    V[0] = 0
    # P[i] stocke l'index du prédécesseur
    P = [0] * (n + 1)

    # 4. Algorithme de Split (Découpage)
    for i in range(n):
        if V[i] == float('inf'):
            continue 
        
        load = 0
        current_time = 0
        current_location = depot
        route_cost = 0
        
        for j in range(i + 1, n + 1):
            client = giant_tour[j - 1]
            
            # Vérification Capacité
            load += client.demand
            if load > vehicle_capacity:
                break 
            
            # Vérification Temps
            travel_time = math.hypot(current_location.x - client.x, current_location.y - client.y)
            arrival_time = current_time + travel_time
            
            if arrival_time > client.due_time:
                break 
            
            current_time = max(arrival_time, client.ready_time) + client.service_time
            route_cost += travel_time
            current_location = client
            
            # Coût total si on rentre au dépôt
            return_cost = math.hypot(current_location.x - depot.x, current_location.y - depot.y)
            total_route_cost = route_cost + return_cost
            
            # Relaxation (Plus court chemin)
            if V[i] + total_route_cost < V[j]:
                V[j] = V[i] + total_route_cost
                P[j] = i 
                
    # 5. Reconstruire les routes
    if V[n] == float('inf'):
        print("ATTENTION: Le tour géant n'a pas pu être découpé. Ajustez alpha et beta !")
        return []

    routes = []
    curr = n
    while curr > 0:
        prev = P[curr]
        routes.append(giant_tour[prev:curr])
        curr = prev
        
    routes.reverse() 
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

    filename = "./Archive/c101.txt"
    depot, customers, vehicle_capacity = read_solomon_file(filename)

    #routes = initial_solution_sweep_split(customers, depot, vehicle_capacity)
    routes = initial_solution_hybrid_split(customers, depot, vehicle_capacity, alpha=0.5, beta=0.5)
    print("Number of routes:", len(routes))

    total_distance = 0

    for i, route in enumerate(routes):
        load = sum(c.demand for c in route)
        print(f"Route {i+1} | Load {load} :", [c.id for c in route])

        current_location = depot
        route_distance = 0
        for c in route:
            route_distance += distance(current_location, c)
            current_location = c
        route_distance += distance(current_location, depot)
        print(f"  Distance: {route_distance:.2f}")
        total_distance += route_distance

    print("Total distance:", total_distance)

    