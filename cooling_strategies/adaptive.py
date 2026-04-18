import math
import random

class TemperatureSchedule:
    def __init__(self, T_init, T_min, alpha=0.995,
                 target_acceptance=0.2, adapt_interval=500,
                 reheat_factor=1.5, reheat_patience=2000):
        self.T = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.target_acc = target_acceptance
        self.adapt_interval = adapt_interval
        self.reheat_factor = reheat_factor
        self.reheat_patience = reheat_patience
        self._iter = 0
        self._accepted_window = 0
        self._total_window = 0
        self._no_improve_count = 0

    def cool(self):
        self.T = max(self.T_min, self.T * self.alpha)
        self._iter += 1

    def record(self, accepted: bool, improved: bool):
        self._total_window += 1
        if accepted:
            self._accepted_window += 1
        if not improved:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        if self._total_window >= self.adapt_interval:
            acc_rate = self._accepted_window / self._total_window
            if acc_rate < self.target_acc * 0.5:
                self.alpha = min(0.9999, self.alpha * 1.005)
            elif acc_rate > self.target_acc * 2.0:
                self.alpha = max(0.990, self.alpha * 0.995)
            self._accepted_window = 0
            self._total_window = 0

        if self._no_improve_count >= self.reheat_patience:
            self.T = min(self.T * self.reheat_factor, self.T * 5)
            self._no_improve_count = 0

    def accept(self, delta: float) -> bool:
        if delta < 0:
            return True
        if self.T <= 0:
            return False
        return random.random() < math.exp(-delta / self.T)