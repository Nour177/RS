import math

def lire_fichier_vrptw(chemin_fichier):
    coordonnees = []
    demandes = []
    fenetres_temps = []
    temps_service = []
    capacite_vehicule = 0
    
    with open(chemin_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        
    lecture_clients = False
    
    for ligne in lignes:
        # Split the line by whitespace
        elements = ligne.split()
        
        # Skip empty lines
        if not elements:
            continue
            
        # Extract vehicle capacity (The line with "25 200" under VEHICLE)
        # We check if there are exactly 2 numbers
        if len(elements) == 2 and elements[0].isdigit() and elements[1].isdigit():
            capacite_vehicule = int(elements[1])
            
        # Detect the start of the customer data (Customer 0 is the depot)
        if elements[0] == '0' and len(elements) >= 7:
            lecture_clients = True
            
        # Once we reach the data section, parse every line
        if lecture_clients:
            # elements = [CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME]
            x = float(elements[1])
            y = float(elements[2])
            demande = int(elements[3])
            ready_time = float(elements[4])
            due_date = float(elements[5])
            service = float(elements[6])
            
            # Since CUST NO. is sequential from 0, appending naturally aligns the indices
            coordonnees.append((x, y))
            demandes.append(demande)
            fenetres_temps.append((ready_time, due_date))
            temps_service.append(service)
            
    return coordonnees, demandes, fenetres_temps, temps_service, capacite_vehicule


def calculer_matrice_distances(coordonnees):

    n = len(coordonnees)
    # Initialize a 2D list (matrix) filled with 0.0
    matrice_distances = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coordonnees[i]
                x2, y2 = coordonnees[j]
                
                # Calcul de la distance euclidienne
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                matrice_distances[i][j] = distance
                
    return matrice_distances