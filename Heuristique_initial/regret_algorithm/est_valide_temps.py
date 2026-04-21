# --- Helper Function: Validate Time Windows ---
def est_valide_temps(route_proposee, fenetres_temps, temps_service, matrice_distances):
    temps_actuel = 0
    for i in range(len(route_proposee) - 1):
        noeud_courant = route_proposee[i]
        noeud_suivant = route_proposee[i+1]
        
        # Check if we arrived too late at the current node
        if temps_actuel > fenetres_temps[noeud_courant][1]:
            return False 
        
        # If we arrive too early, we must wait until the 'early_start' time
        temps_actuel = max(temps_actuel, fenetres_temps[noeud_courant][0])
        
        # Add the service time (loading/unloading)
        temps_actuel += temps_service[noeud_courant]
        
        # Add travel time to the next node
        temps_actuel += matrice_distances[noeud_courant][noeud_suivant]
        
    # Check if we arrive back at the depot (or last node) too late
    dernier_noeud = route_proposee[-1]
    if temps_actuel > fenetres_temps[dernier_noeud][1]:
        return False
        
    return True
# ----------------------------------------------
