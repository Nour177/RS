def trouver_meilleur_seed(depot, clients_non_assignes, matrice_distances, fenetres_temps, alpha=1.0, beta=0.5, gamma=0.5):
# Si tu mets alpha = 1 et beta = 0, gamma = 0, le camion ira toujours au client le plus éloigné.
# Si tu mets alpha = 0 et beta = 1, le camion ira toujours chez le client qui ouvre le plus tôt.
# Si tu mets alpha = 0 et gamma = 1, le camion ira toujours chez le client avec la fenêtre de temps la plus serrée.

    """
    Trouve le meilleur client pour initialiser une route en combinant 
    l'éloignement spatial et la criticité temporelle.
    
    Poids (hyperparamètres) à ajuster selon l'importance relative de chaque facteur :
    - alpha : Importance de la distance
    - beta  : Importance de commencer tôt
    - gamma : Importance d'une fenêtre de temps très courte
    """
    if not clients_non_assignes:
        return None
        
    meilleur_client = None
    meilleur_score = -float('inf') # assuming higher score is better
    
    for client in clients_non_assignes:
        # 1. Distance depuis le dépôt
        distance = matrice_distances[depot][client]
        
        # 2. Informations sur le temps
        ready_time = fenetres_temps[client][0]
        due_date = fenetres_temps[client][1]
        largeur_fenetre = due_date - ready_time
        
        # Calcul du score de criticité
        score = (alpha * distance) - (beta * ready_time) - (gamma * largeur_fenetre)
        
        if score > meilleur_score:
            meilleur_score = score
            meilleur_client = client
            
    return meilleur_client