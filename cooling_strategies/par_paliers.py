import math
import random
class StepSchedule:
    def __init__(self, T_init, T_min, alpha=0.90, longueur_palier=100):
        self.T = T_init
        self.T_min = T_min
        self.alpha = alpha  # Baisse plus forte que la version classique (ex: 0.90 au lieu de 0.99)
        self.longueur_palier = longueur_palier
        self.etape_actuelle = 0

    def cool(self):
        self.etape_actuelle += 1
        # On ne baisse la température que si on a fini le palier
        if self.etape_actuelle >= self.longueur_palier:
            self.T = max(self.T_min, self.T * self.alpha)
            self.etape_actuelle = 0  # On réinitialise le compteur pour le prochain palier

    def record(self, accepted: bool, improved: bool):
        # Stratégie non adaptative, on l'ignore
        pass

    def accept(self, delta: float) -> bool:
        if delta < 0:
            return True
        if self.T <= 0:
            return False
        return random.random() < math.exp(-delta / self.T)