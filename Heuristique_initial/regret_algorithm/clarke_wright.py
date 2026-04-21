from .est_valide_temps import est_valide_temps 


def generer_solution_clarke_wright(tous_les_clients, depot, matrice_distances, capacite_max, demandes, fenetres_temps, temps_service):
    """
    Génère une solution initiale en utilisant l'algorithme de Clarke & Wright (Savings).
    """
    # ÉTAPE 1 : Initialisation (Une route par client)
    # routes_actives est un dictionnaire pour retrouver vite la route d'un client
    routes = []
    for client in tous_les_clients:
        nouvelle_route = [depot, client, depot]
        routes.append(nouvelle_route)
        
    # ÉTAPE 2 : Calcul de toutes les économies (AMÉLIORÉ AVEC LE TEMPS)
    economies = []
    gamma = 1.0 # Poids de la pénalité de temps (à ajuster, ex: 0.5, 1.0, 2.0)
    
    for i in tous_les_clients:
        for j in tous_les_clients:
            if i != j:
                # 1. Économie spatiale classique
                economie_dist = matrice_distances[i][depot] + matrice_distances[depot][j] - matrice_distances[i][j]
                
                # 2. Calcul de la pénalité temporelle (Temps d'attente)
                fin_service_i = fenetres_temps[i][0] + temps_service[i] # Le plus tôt où on part de i
                arrivee_chez_j = fin_service_i + matrice_distances[i][j]
                ouverture_j = fenetres_temps[j][0]
                
                # S'il arrive avant l'ouverture, il attend. Sinon, l'attente est 0.
                temps_attente = max(0, ouverture_j - arrivee_chez_j)
                
                # 3. Nouvelle économie combinée
                economie_finale = economie_dist - (gamma * temps_attente)
                
                # On ne garde que les fusions qui ont un vrai sens (économie > 0)
                if economie_finale > 0:
                    economies.append((economie_finale, i, j))
                    
    # ÉTAPE 3 : Trier les économies de la plus grande à la plus petite
    economies.sort(key=lambda x: x[0], reverse=True)
    
    # ÉTAPE 4 : Processus de Fusion (Merging)
    for economie, i, j in economies:
        # Trouver la route actuelle qui contient i et celle qui contient j
        route_i = next((r for r in routes if i in r), None)
        route_j = next((r for r in routes if j in r), None)
        
        # Conditions pour pouvoir fusionner :
        # 1. i et j ne sont pas déjà dans la même route
        # 2. i doit être le DERNIER client de sa route (juste avant le retour au dépôt)
        # 3. j doit être le PREMIER client de sa route (juste après le départ du dépôt)
        if route_i and route_j and route_i != route_j:
            if route_i[-2] == i and route_j[1] == j:
                
                # Créer la route fusionnée candidate
                # Exemple : [0, A, B, 0] et [0, C, D, 0] devient [0, A, B, C, D, 0]
                route_fusionnee = route_i[:-1] + route_j[1:]
                
                # Vérifier la contrainte de Capacité
                charge_totale = sum(demandes[client] for client in route_fusionnee if client != depot)
                
                if charge_totale <= capacite_max:
                    # Vérifier la contrainte des Fenêtres de temps
                    if est_valide_temps(route_fusionnee, fenetres_temps , temps_service, matrice_distances):
                        # La fusion est validée ! On remplace les deux petites routes par la grande
                        routes.remove(route_i)
                        routes.remove(route_j)
                        routes.append(route_fusionnee)
                        
    return routes