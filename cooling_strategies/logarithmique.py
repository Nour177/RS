import math
import random

class LogarithmicSchedule:
    def __init__(self, T_init, T_min):
        self.T_init = T_init
        self.T = T_init
        self.T_min = T_min
        self.k = 1  # Compteur d'itérations

    def cool(self):
        self.k += 1
        # On applique la formule logarithmique
        nouvelle_temp = self.T_init / math.log(1 + self.k)
        self.T = max(self.T_min, nouvelle_temp)

    def record(self, accepted: bool, improved: bool):
        # Cette stratégie n'est pas adaptative, on ignore simplement l'historique
        pass

    def accept(self, delta: float) -> bool:
        if delta < 0:
            return True
        if self.T <= 0:
            return False
        return random.random() < math.exp(-delta / self.T)