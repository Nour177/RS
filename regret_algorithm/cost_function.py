import time
from read_file import lire_fichier_vrptw, calculer_matrice_distances
from algo import generer_solution_initiale_randomisee
from clarke_wright import generer_solution_clarke_wright

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale parcourue par tous les camions."""
    distance_totale = 0.0
    for route in solution:
        for i in range(len(route) - 1):
            noeud_actuel = route[i]
            noeud_suivant = route[i+1]
            distance_totale += matrice_distances[noeud_actuel][noeud_suivant]
    return distance_totale


def evaluer_performances(chemin_fichier, algorithm, nb_executions=10, K=3, alpha=1.0, beta=0.5, gamma=0.5):
    print(f"--- Évaluation sur {chemin_fichier} ({nb_executions} exécutions) ---")
    
    # 1. Préparation des données (Temps non inclus dans le chrono de l'algorithme)
    coordonnees, demandes, fenetres_temps, temps_service, capacite_max = lire_fichier_vrptw(chemin_fichier)
    matrice_distances = calculer_matrice_distances(coordonnees)
    depot = 0
    # exclude depot from clients list for insertion
    tous_les_clients = list(range(1, len(coordonnees)))
    
    # Variables pour les statistiques
    historique_vehicules = []
    historique_distances = []
    
    meilleure_solution = None
    min_vehicules_trouve = float('inf')
    min_distance_trouvee = float('inf')
    
    # 2. Lancement du chronomètre
    temps_debut = time.time()
    
    # 3. Boucle d'exécutions multiples
    for i in range(nb_executions):
        # On appelle ton algorithme (Assure-toi de lui passer alpha, beta, gamma si tu les as intégrés)
        solution = algorithm(
            tous_les_clients, depot, matrice_distances, 
            capacite_max, demandes, fenetres_temps, temps_service
        )
        
        # Évaluation de cette exécution
        nb_vehicules = len(solution)
        distance = calculer_distance_totale(solution, matrice_distances)
        
        historique_vehicules.append(nb_vehicules)
        historique_distances.append(distance)
        
        # Mise à jour de la "Meilleure Solution Globale"
        # On priorise toujours le nombre de véhicules, puis la distance
        if nb_vehicules < min_vehicules_trouve or (nb_vehicules == min_vehicules_trouve and distance < min_distance_trouvee):
            min_vehicules_trouve = nb_vehicules
            min_distance_trouvee = distance
            meilleure_solution = solution
            
    # 4. Fin du chronomètre
    temps_fin = time.time()
    temps_total_execution = temps_fin - temps_debut
    temps_moyen_par_execution = temps_total_execution / nb_executions
    
    # 5. Calcul des moyennes
    moyenne_vehicules = sum(historique_vehicules) / nb_executions
    moyenne_distances = sum(historique_distances) / nb_executions
    
    # Affichage des résultats
    print(f"Temps total       : {temps_total_execution:.4f} secondes (soit {temps_moyen_par_execution:.4f} s/exec)")
    print(f"Véhicules utilisés: Min = {min(historique_vehicules)}, Moyenne = {moyenne_vehicules:.2f}, Max = {max(historique_vehicules)}")
    print(f"Distance totale   : Min = {min(historique_distances):.2f}, Moyenne = {moyenne_distances:.2f}, Max = {max(historique_distances):.2f}")
    print(f"Meilleure solution trouvée: {meilleure_solution} avec {min_vehicules_trouve} véhicules et distance {min_distance_trouvee:.2f}")
    print("-" * 50)
    
    return {
        "Min Vehicules": min(historique_vehicules),
        "Moyenne Vehicules": moyenne_vehicules,
        "Max Vehicules": max(historique_vehicules),
        "Min Distance": min(historique_distances),
        "Moyenne Distance": moyenne_distances,
        "Max Distance": max(historique_distances),
        "Temps Moyen (s)": temps_moyen_par_execution,
        "Historique Distances": historique_distances # Utile pour la boîte à moustaches
    }

evaluer_performances('Archive/C101.txt',generer_solution_initiale_randomisee)
evaluer_performances('Archive/C101.txt',generer_solution_clarke_wright)