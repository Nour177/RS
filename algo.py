import random
from seeds import trouver_meilleur_seed
from est_valide_temps import est_valide_temps

def generer_solution_initiale_randomisee(tous_les_clients, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service, K=3):
    """
    Generates a randomized initial VRP solution using a Regret-K heuristic.
    """
    routes_finales = []
    # Copy the list so we don't modify the original
    clients_non_assignes = list(tous_les_clients) 
    
    # Loop as long as there are clients left to route
    while clients_non_assignes:
        # Create an empty route: [depot, depot]
        nouvelle_route = [depot, depot]
        
        # --- 1. SEED STEP (Initialize the truck) ---
        client_seed = trouver_meilleur_seed(depot, clients_non_assignes, matrice_distances, fenetres_temps)
        if client_seed is not None:
            # Insert between the start and end depots
            nouvelle_route.insert(1, client_seed)
            clients_non_assignes.remove(client_seed)
        
        route_pleine = False
        
        # --- 2. RANDOMIZED REGRET STEP ---
        while not route_pleine and clients_non_assignes:
            candidats_valides = []
            
            for client in clients_non_assignes:
                # We pass [nouvelle_route] as a list because our functions expect a list of routes
                pos_1, cout_1 = trouver_meilleure_insertion(client, nouvelle_route, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service)
                pos_2, cout_2 = trouver_deuxieme_meilleure_insertion(client, nouvelle_route, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service)
                
                # If pos_1[0] == -1, it means the client doesn't fit in the current route 
                # (violates capacity or time windows) and requires a new truck.
                if pos_1 is None or pos_1 == -1:
                    continue # Skip to the next client
                    
                # Calculate regret. If no 2nd position exists (cout_2 is infinity), 
                # the regret is infinite (we MUST place it here or it won't fit anywhere else in this truck).
                regret = float('inf') if cout_2 == float('inf') else cout_2 - cout_1

                # 3. Intégration de la fenêtre de temps
                ready_time = fenetres_temps[client][0]
                due_date = fenetres_temps[client][1]
                largeur_fenetre = due_date - ready_time
                
                # hyperparamètres alpha et gamma pour équilibrer l'importance de la distance et du temps
                alpha = 1.0  # Poids du regret de distance
                gamma = 0.5  # Poids de l'urgence de la fenêtre de temps
                
                # Si le regret est infini, on le garde infini, sinon on applique la formule
                if regret == float('inf'):
                    regret_final = float('inf')
                else:
                    regret_final = (alpha * regret) - (gamma * largeur_fenetre)
                
                # Save the valid candidate
                candidats_valides.append({
                    'client': client, 
                    'position': pos_1, # insert_index for this route
                    'regret': regret_final
                })
                
            # --- 3. RANDOMIZED INSERTION DECISION ---
            if not candidats_valides:
                # No more clients can fit into this truck
                route_pleine = True
            else:
                # Sort descending by regret
                candidats_valides.sort(key=lambda x: x['regret'], reverse=True)
                
                # Isolate the top K candidates
                limite = min(K, len(candidats_valides))
                top_k_candidats = candidats_valides[:limite]
                
                # Randomly pick one of the top K
                candidat_choisi = random.choice(top_k_candidats)
                client_a_inserer = candidat_choisi['client']
                
                # Insert the chosen client into the route. 
                # pos_1 is (route_idx, insert_idx). Since we only passed one route, route_idx is always 0.
                index_insertion = candidat_choisi['position']
                nouvelle_route.insert(index_insertion, client_a_inserer)
                
                # Remove from waiting list
                clients_non_assignes.remove(client_a_inserer)
                
        # The truck is full or blocked, add it to the final schedule
        routes_finales.append(nouvelle_route)
        
    return routes_finales

def trouver_deuxieme_meilleure_insertion(client, route, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service):
    """
    Finds the second best insertion for a client into current route, respecting capacity and time windows.
    
    Returns:
        tuple: (index_insertion, second_best_cost)
               Returns (None, float('inf')) if no second valid insertion exists.
    """
    POIDS_CAMION = 100000  # Massive penalty to prioritize minimizing trucks
    
    # Track the top 2 best insertions
    cout_1 = float('inf')
    pos_1 = None
    
    cout_2 = float('inf')
    pos_2 = None
    

    # --- 1. Evaluate inserting into EXISTING routes ---

    charge_actuelle = sum(demandes[c] for c in route if c != depot)
    
    if charge_actuelle + demandes[client] <= capacite_max:
        
        for j in range(1, len(route)):
            route_test = route[:j] + [client] + route[j:]
            
            if est_valide_temps(route_test, fenetres_temps, temps_service, matrice_distances):
                noeud_precedent = route[j-1]
                noeud_suivant = route[j]
                
                cout = (matrice_distances[noeud_precedent][client] + 
                        matrice_distances[client][noeud_suivant] - 
                        matrice_distances[noeud_precedent][noeud_suivant])
                
                # Update Top 2 Logic
                if cout < cout_1:
                    # Demote 1st to 2nd
                    cout_2 = cout_1
                    pos_2 = pos_1
                    # Update 1st
                    cout_1 = cout
                    pos_1 = j
                elif cout < cout_2:
                    # Update 2nd only
                    cout_2 = cout
                    pos_2 = j
                        
    # --- 2. Evaluate creating a NEW route ---
    route_nouveau_camion = [depot, client, depot]
    
    if est_valide_temps(route_nouveau_camion, fenetres_temps, temps_service, matrice_distances):
        dist_nouvelle_route = matrice_distances[depot][client] + matrice_distances[client][depot]
        cout_nouveau_camion = POIDS_CAMION + dist_nouvelle_route
        
        # Update Top 2 Logic for the new truck scenario
        if cout_nouveau_camion < cout_1:
            cout_2 = cout_1
            pos_2 = pos_1
            cout_1 = cout_nouveau_camion
            pos_1 = -1
        elif cout_nouveau_camion < cout_2:
            cout_2 = cout_nouveau_camion
            pos_2 = -1
            
    # Return the second best (which might be None/inf if only 1 or 0 valid spots exist)
    return pos_2, cout_2

def trouver_meilleure_insertion(client, route, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service):
    """
    Finds the best insertion for a client into routes, respecting capacity and time windows.
    
    Args:
        client: The ID of the client to insert.
        route: The current route into which to insert the client.
        depot: The ID representing the depot.
        matrice_distances: 2D array of distances.
        capacite_max: Maximum capacity of a truck.
        demandes: Dict mapping client ID -> demand.
        fenetres_temps: Dict mapping node ID -> [early_start, late_end].
        temps_service: Dict mapping node ID -> time spent at the client location.
        
    Returns:
        tuple: ((index_route, index_insertion), best_cost)
    """
    POIDS_CAMION = 100000  # Massive penalty to prioritize minimizing trucks
    
    meilleur_cout = float('inf')
    meilleure_position = None  # (index_route, index_insertion)
    
    # --- 1. Evaluate inserting into EXISTING routes ---
    
    charge_actuelle = sum(demandes[c] for c in route if c != depot)
    
    # Check Capacity First (it's faster to compute than time)
    if charge_actuelle + demandes[client] <= capacite_max:
        
        for j in range(1, len(route)):
            # Create a temporary route to test the timeline
            route_test = route[:j] + [client] + route[j:]
            
            # Check Time Windows
            if est_valide_temps(route_test, fenetres_temps, temps_service, matrice_distances):
                
                # Calculate distance cost if constraints are met
                noeud_precedent = route[j-1]
                noeud_suivant = route[j]
                
                dist_ajoutee = (matrice_distances[noeud_precedent][client] + 
                                matrice_distances[client][noeud_suivant] - 
                                matrice_distances[noeud_precedent][noeud_suivant])
                
                if dist_ajoutee < meilleur_cout:
                    meilleur_cout = dist_ajoutee
                    meilleure_position = j
                        
    # --- 2. Evaluate creating a NEW route ---
    route_nouveau_camion = [depot, client, depot]
    
    # We must ensure the client can actually be serviced by a brand new truck 
    # (e.g., they aren't so far away that getting there immediately breaks the window)
    if est_valide_temps(route_nouveau_camion, fenetres_temps, temps_service, matrice_distances):
        dist_nouvelle_route = matrice_distances[depot][client] + matrice_distances[client][depot]
        cout_nouveau_camion = POIDS_CAMION + dist_nouvelle_route
        
        if cout_nouveau_camion < meilleur_cout:
            meilleur_cout = cout_nouveau_camion
            meilleure_position = -1  # Special marker for "new truck"
            
    return meilleure_position, meilleur_cout