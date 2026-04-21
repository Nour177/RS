import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Heuristique_initial.regret_algorithm.cost_function import evaluer_performances
from Heuristique_initial.regret_algorithm.clarke_wright import generer_solution_clarke_wright
from Heuristique_initial.regret_algorithm.algo import generer_solution_initiale_randomisee

def analyser_dossier_et_visualiser(dossier_instances):
    resultats_globaux = []
    donnees_brutes_boxplots = [] # Pour stocker chaque exécution individuelle

    # 1. Parcours de tous les fichiers du dossier
    for fichier in os.listdir(dossier_instances):
        if not fichier.endswith('.txt'):
            continue
            
        chemin_fichier = os.path.join(dossier_instances, fichier)
        
        # Détermination de la catégorie (C, R, ou RC)
        if fichier.startswith('RC'):
            categorie = 'RC (Random-Clustered)'
        elif fichier.startswith('R'):
            categorie = 'R (Random)'
        elif fichier.startswith('C'):
            categorie = 'C (Clustered)'
        else:
            categorie = 'Autre'

        # --- Exécution Clarke & Wright (Déterministe -> 1 seule exécution) ---
        stats_cw = evaluer_performances(chemin_fichier, generer_solution_clarke_wright, nb_executions=1)
        
        resultats_globaux.append({
            "Fichier": fichier,
            "Catégorie": categorie,
            "Algorithme": "Clarke & Wright",
            "Distance Moyenne": stats_cw["Moyenne Distance"],
            "Véhicules Moyens": stats_cw["Moyenne Vehicules"],
            "Temps Moyen (s)": stats_cw["Temps Moyen (s)"]
        })
        
        donnees_brutes_boxplots.append({
            "Catégorie": categorie,
            "Algorithme": "Clarke & Wright",
            "Distance": stats_cw["Moyenne Distance"] # Une seule valeur
        })

        # --- Exécution Regret + Randomisé (Stochastique -> 10 exécutions) ---
        stats_regret = evaluer_performances(chemin_fichier, generer_solution_initiale_randomisee, nb_executions=10)
        
        resultats_globaux.append({
            "Fichier": fichier,
            "Catégorie": categorie,
            "Algorithme": "Regret Randomisé",
            "Distance Moyenne": stats_regret["Moyenne Distance"],
            "Véhicules Moyens": stats_regret["Moyenne Vehicules"],
            "Temps Moyen (s)": stats_regret["Temps Moyen (s)"]
        })
        
        # Ajout des 10 exécutions individuelles pour le boxplot
        for dist in stats_regret["Historique Distances"]:
            donnees_brutes_boxplots.append({
                "Catégorie": categorie,
                "Algorithme": "Regret Randomisé",
                "Distance": dist
            })

    # 2. Conversion en DataFrames Pandas
    df_resultats = pd.DataFrame(resultats_globaux)
    df_boxplots = pd.DataFrame(donnees_brutes_boxplots)

    # Affichage d'un tableau récapitulatif dans la console
    print("\n=== RÉSUMÉ DES PERFORMANCES PAR CATÉGORIE ===")
    resume = df_resultats.groupby(['Catégorie', 'Algorithme'])[['Distance Moyenne', 'Véhicules Moyens', 'Temps Moyen (s)']].mean()
    print(resume)

    # 3. GÉNÉRATION DES GRAPHIQUES
    sns.set_theme(style="whitegrid")
    
    # Graphique 1 : Diagramme en barres (Distance Moyenne par Catégorie)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultats, x="Catégorie", y="Distance Moyenne", hue="Algorithme", errorbar=None)
    plt.title("Comparaison des Distances Moyennes par Type d'Instance")
    plt.ylabel("Distance Totale Moyenne")
    plt.tight_layout()
    plt.show()

    # Graphique 1 : Diagramme en barres (Distance Moyenne par Catégorie)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultats, x="Catégorie", y="Véhicules Moyens", hue="Algorithme", errorbar=None)
    plt.title("Comparaison des Véhicules Moyens par Type d'Instance")
    plt.ylabel("Nombre de Véhicules Moyens")
    plt.tight_layout()
    plt.show()

    # Graphique 2 : Boîte à moustaches (Stabilité)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_boxplots, x="Catégorie", y="Distance", hue="Algorithme")
    plt.title("Distribution des Distances (Stabilité de l'algorithme Randomisé vs C&W)")
    plt.ylabel("Distance Totale (par exécution)")
    plt.tight_layout()
    plt.show()

    # Graphique 3 : Nuage de points (Temps vs Qualité)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_resultats, x="Temps Moyen (s)", y="Distance Moyenne", 
                    hue="Algorithme", style="Catégorie", s=100)
    plt.title("Compromis Temps d'exécution vs Qualité de la solution")
    plt.xlabel("Temps Moyen par exécution (secondes)")
    plt.ylabel("Distance Totale Moyenne")
    plt.tight_layout()
    plt.show()

    # Graphique 3 : Nuage de points (Temps vs Qualité)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_resultats, x="Véhicules Moyens", y="Distance Moyenne", 
                    hue="Algorithme", style="Catégorie", s=100)
    plt.title("Compromis Temps d'exécution vs Qualité de la solution")
    plt.xlabel("Véhicules Moyens")
    plt.ylabel("Distance Totale Moyenne")
    plt.tight_layout()
    plt.show()

# Pour lancer l'analyse (remplace 'Archive' par le bon chemin si besoin)
analyser_dossier_et_visualiser('Archive')