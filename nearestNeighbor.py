import math

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


# Initial solution (Nearest Neighbor)
def initial_solution_nearest_neighbor(clients, depot, vehicle_capacity):
    unvisited = clients[:]
    routes = []

    while unvisited:
        route = []
        load = 0
        current_time = 0
        current_location = depot

        while True:
            feasible_clients = []

            for client in unvisited:
                if load + client.demand > vehicle_capacity:
                    continue

                travel_time = distance(current_location, client)
                arrival_time = current_time + travel_time

                if arrival_time > client.due_time:
                    continue

                feasible_clients.append((client, travel_time))

            if not feasible_clients:
                break

            next_client = min(feasible_clients, key=lambda x: x[1])[0]

            travel_time = distance(current_location, next_client)
            arrival_time = current_time + travel_time

            current_time = max(arrival_time, next_client.ready_time) + next_client.service_time
            load += next_client.demand

            route.append(next_client)
            unvisited.remove(next_client)
            current_location = next_client

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

    filename = "./Archive/C107.txt"
    depot, customers, vehicle_capacity = read_solomon_file(filename)

    routes = initial_solution_nearest_neighbor(customers, depot, vehicle_capacity)

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