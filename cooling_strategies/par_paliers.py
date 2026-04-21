import math
import random
class StepSchedule:
    def __init__(self, T_init, T_min, max_iter=50000, longueur_palier=200,alpha=None):
        self.T = T_init
        self.T_min = T_min
        self.longueur_palier = longueur_palier
        self.etape_actuelle = 0
        
        # Calcul dynamique du alpha parfait
        nb_paliers = max_iter // longueur_palier
        if nb_paliers > 0:
            self.alpha = (T_min / T_init) ** (1 / nb_paliers)
        else:
            self.alpha = 0.99 # Fallback de sécurité

    def cool(self):
        self.etape_actuelle += 1
        if self.etape_actuelle >= self.longueur_palier:
            self.T = max(self.T_min, self.T * self.alpha)
            self.etape_actuelle = 0

    def record(self, accepted: bool, improved: bool):
        pass

    def accept(self, delta: float) -> bool:
        if delta < 0: return True
        if self.T <= 0: return False
        return random.random() < math.exp(-delta / self.T)