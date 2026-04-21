
import math
import random
class LogarithmicSchedule:
    def __init__(self, T_init, T_min, max_iter=50000,cooling_factor=1.0):
        self.T_init = T_init
        self.T = T_init
        self.T_min = T_min
        self.k = 1
        
        # Facteur de compression pour atteindre T_min à max_iter
        # On résout T_min = T_init / log(e + beta * max_iter)
        cible_log = T_init / T_min
        self.beta = (math.exp(cible_log) - math.e) / max_iter if cible_log < 50 else 1.0

    def cool(self):
        self.k += 1
        # Formule logarithmique bornée pour éviter les divisions par zéro ou les chutes trop lentes
        nouvelle_temp = self.T_init / math.log(math.e + self.beta * self.k)
        self.T = max(self.T_min, nouvelle_temp)

    def record(self, accepted: bool, improved: bool):
        pass

    def accept(self, delta: float) -> bool:
        if delta < 0: return True
        if self.T <= 0: return False
        try:
            return random.random() < math.exp(-delta / self.T)
        except OverflowError: # Sécurité si delta/T est trop grand
            return False