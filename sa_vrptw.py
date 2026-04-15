"""
=============================================================
  Recuit Simulé (Simulated Annealing) pour le VRPTW
=============================================================
Auteur : votre partie du projet
Interface : reçoit une solution initiale sous forme de
            liste de routes  [[0,...,0], [0,...,0], ...]
            (dépôt = nœud 0, inclus au début et à la fin)

Benchmark cible : Solomon (100 clients) – facilement
                  adaptable à Homberger.
=============================================================
"""

import math
import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  1.  STRUCTURES DE DONNÉES
# ─────────────────────────────────────────────────────────────

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float   # a_i  (fenêtre de temps début)
    due_date: float     # b_i  (fenêtre de temps fin)
    service_time: float # s_i


@dataclass
class Instance:
    """Contient toutes les données du problème."""
    depot: Customer
    customers: List[Customer]          # indexés 1..n
    vehicle_capacity: float
    num_vehicles: int

    def node(self, idx: int) -> Customer:
        """Retourne le nœud (0 = dépôt, 1..n = client)."""
        if idx == 0:
            return self.depot
        return self.customers[idx - 1]

    def distance(self, i: int, j: int) -> float:
        a, b = self.node(i), self.node(j)
        return math.hypot(a.x - b.x, a.y - b.y)


# ─────────────────────────────────────────────────────────────
#  2.  ÉVALUATION D'UNE SOLUTION
# ─────────────────────────────────────────────────────────────

def evaluate_route(route: List[int], inst: Instance
                   ) -> Tuple[float, float, bool]:
    """
    Évalue une route.
    Retourne (distance_totale, pénalité_contraintes, faisable).
    route : [0, c1, c2, ..., ck, 0]
    """
    dist = 0.0
    load = 0.0
    time_now = 0.0
    penalty = 0.0
    feasible = True

    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        d = inst.distance(i, j)
        dist += d
        time_now += d

        if j != 0:
            cust = inst.node(j)
            load += cust.demand
            # Attente si on arrive trop tôt
            if time_now < cust.ready_time:
                time_now = cust.ready_time
            # Violation fenêtre de temps
            if time_now > cust.due_date:
                viol = time_now - cust.due_date
                penalty += viol
                feasible = False
            time_now += cust.service_time

    # Violation capacité
    if load > inst.vehicle_capacity:
        penalty += (load - inst.vehicle_capacity) * 1000
        feasible = False

    return dist, penalty, feasible


def total_cost(solution: List[List[int]], inst: Instance,
               penalty_weight: float = 1000.0) -> float:
    """Coût total = distance + poids × violations."""
    total = 0.0
    for route in solution:
        d, p, _ = evaluate_route(route, inst)
        total += d + penalty_weight * p
    return total


def is_feasible(solution: List[List[int]], inst: Instance) -> bool:
    return all(evaluate_route(r, inst)[2] for r in solution)


# ─────────────────────────────────────────────────────────────
#  3.  OPÉRATEURS DE VOISINAGE
# ─────────────────────────────────────────────────────────────

# ---------- 3a. Relocate (inter-route) ----------
def neighbor_relocate(solution: List[List[int]]
                      ) -> Optional[List[List[int]]]:
    """
    Retire un client d'une route et le réinsère dans une autre
    (ou ailleurs dans la même route) à la meilleure position.
    """
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 2]
    if not routes_with_clients:
        return None

    r1_idx = random.choice(routes_with_clients)
    route1 = sol[r1_idx]
    # Position du client à déplacer (hors dépôts)
    pos = random.randint(1, len(route1) - 2)
    client = route1[pos]

    # Retirer le client
    new_r1 = route1[:pos] + route1[pos+1:]
    if len(new_r1) == 2:           # route vide → supprimer
        sol.pop(r1_idx)
        r2_candidates = list(range(len(sol)))
    else:
        sol[r1_idx] = new_r1
        r2_candidates = list(range(len(sol)))

    if not r2_candidates:
        return None

    r2_idx = random.choice(r2_candidates)
    route2 = sol[r2_idx]
    ins_pos = random.randint(1, len(route2) - 1)
    sol[r2_idx] = route2[:ins_pos] + [client] + route2[ins_pos:]
    return sol


# ---------- 3b. Swap (inter-route) ----------
def neighbor_swap(solution: List[List[int]]
                  ) -> Optional[List[List[int]]]:
    """Échange deux clients de routes différentes."""
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 2]
    if len(routes_with_clients) < 2:
        return None

    r1_idx, r2_idx = random.sample(routes_with_clients, 2)
    r1, r2 = sol[r1_idx], sol[r2_idx]

    p1 = random.randint(1, len(r1) - 2)
    p2 = random.randint(1, len(r2) - 2)

    r1[p1], r2[p2] = r2[p2], r1[p1]
    return sol


# ---------- 3c. 2-opt (intra-route) ----------
def neighbor_2opt(solution: List[List[int]]
                  ) -> Optional[List[List[int]]]:
    """Inversion d'un segment à l'intérieur d'une route."""
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 3]
    if not routes_with_clients:
        return None

    r_idx = random.choice(routes_with_clients)
    route = sol[r_idx]
    n = len(route) - 2   # nombre de clients (hors dépôts)
    if n < 2:
        return None

    i = random.randint(1, len(route) - 3)
    j = random.randint(i + 1, len(route) - 2)
    route[i:j+1] = route[i:j+1][::-1]
    sol[r_idx] = route
    return sol


# ---------- 3d. Or-opt (réinsertion de segment) ----------
def neighbor_or_opt(solution: List[List[int]], seg_len: int = 2
                    ) -> Optional[List[List[int]]]:
    """
    Déplace un segment de `seg_len` clients consécutifs
    vers une autre position (intra ou inter route).
    """
    sol = copy.deepcopy(solution)
    candidates = [i for i, r in enumerate(sol)
                  if len(r) - 2 >= seg_len]
    if not candidates:
        return None

    r1_idx = random.choice(candidates)
    route1 = sol[r1_idx]
    max_start = len(route1) - 1 - seg_len
    if max_start < 1:
        return None

    start = random.randint(1, max_start)
    segment = route1[start:start + seg_len]
    new_r1 = route1[:start] + route1[start + seg_len:]

    if len(new_r1) == 2:
        sol.pop(r1_idx)
        r2_candidates = list(range(len(sol)))
    else:
        sol[r1_idx] = new_r1
        r2_candidates = list(range(len(sol)))

    if not r2_candidates:
        return None

    r2_idx = random.choice(r2_candidates)
    route2 = sol[r2_idx]
    ins_pos = random.randint(1, len(route2) - 1)
    sol[r2_idx] = route2[:ins_pos] + segment + route2[ins_pos:]
    return sol


# ---------- 3e. Cross-route exchange (2 segments) ----------
def neighbor_cross(solution: List[List[int]]
                   ) -> Optional[List[List[int]]]:
    """Échange les fins de deux routes (cross-exchange)."""
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 3]
    if len(routes_with_clients) < 2:
        return None

    r1_idx, r2_idx = random.sample(routes_with_clients, 2)
    r1, r2 = sol[r1_idx], sol[r2_idx]

    cut1 = random.randint(1, len(r1) - 2)
    cut2 = random.randint(1, len(r2) - 2)

    new_r1 = r1[:cut1] + r2[cut2:-1] + [0]
    new_r2 = r2[:cut2] + r1[cut1:-1] + [0]

    sol[r1_idx] = new_r1
    sol[r2_idx] = new_r2
    return sol


# Registre des opérateurs avec leurs poids initiaux
OPERATORS = {
    "relocate":  neighbor_relocate,
    "swap":      neighbor_swap,
    "2opt":      neighbor_2opt,
    "or_opt_1":  lambda s: neighbor_or_opt(s, 1),
    "or_opt_2":  lambda s: neighbor_or_opt(s, 2),
    "or_opt_3":  lambda s: neighbor_or_opt(s, 3),
    "cross":     neighbor_cross,
}


# ─────────────────────────────────────────────────────────────
#  4.  SÉLECTION ADAPTATIVE DES OPÉRATEURS (ROULETTE WHEEL)
# ─────────────────────────────────────────────────────────────

class AdaptiveOperatorSelector:
    """
    Adaptive Large Neighborhood-like weight update :
    chaque opérateur a un score mis à jour selon ses succès.
    """
    def __init__(self, operator_names: List[str],
                 reaction_factor: float = 0.1):
        self.names = operator_names
        self.weights = {n: 1.0 for n in operator_names}
        self.scores  = {n: 0.0 for n in operator_names}
        self.uses    = {n: 0   for n in operator_names}
        self.r = reaction_factor   # vitesse d'adaptation

    def select(self) -> str:
        total = sum(self.weights.values())
        r = random.uniform(0, total)
        cumul = 0.0
        for name, w in self.weights.items():
            cumul += w
            if r <= cumul:
                return name
        return self.names[-1]

    def update(self, name: str, reward: float):
        """reward: 3=nouvelle meilleure, 2=améliorant, 1=accepté, 0=rejeté."""
        self.uses[name] += 1
        self.scores[name] += reward
        # Mise à jour exponentielle du poids
        if self.uses[name] > 0:
            avg = self.scores[name] / self.uses[name]
            self.weights[name] = (1 - self.r) * self.weights[name] + self.r * avg

    def reset_scores(self):
        self.scores = {n: 0.0 for n in self.names}
        self.uses   = {n: 0   for n in self.names}


# ─────────────────────────────────────────────────────────────
#  5.  SCHÉMA DE REFROIDISSEMENT ADAPTATIF
# ─────────────────────────────────────────────────────────────

class TemperatureSchedule:
    """
    Combinaison de :
    - Refroidissement géométrique  T ← α·T
    - Réchauffe automatique si stagnation (reheat)
    - Ajustement de α selon le taux d'acceptation cible
    """
    def __init__(self,
                 T_init: float,
                 T_min: float,
                 alpha: float = 0.995,
                 target_acceptance: float = 0.2,
                 adapt_interval: int = 500,
                 reheat_factor: float = 1.5,
                 reheat_patience: int = 2000):

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

        # ── Adaptation de α ──────────────────────────────────
        if self._total_window >= self.adapt_interval:
            acc_rate = self._accepted_window / self._total_window
            if acc_rate < self.target_acc * 0.5:
                self.alpha = min(0.9999, self.alpha * 1.005)  # refroidir moins vite
            elif acc_rate > self.target_acc * 2.0:
                self.alpha = max(0.990,  self.alpha * 0.995)  # refroidir plus vite
            self._accepted_window = 0
            self._total_window = 0

        # ── Reheat ───────────────────────────────────────────
        if self._no_improve_count >= self.reheat_patience:
            self.T = min(self.T * self.reheat_factor, self.T * 5)
            self._no_improve_count = 0

    def accept(self, delta: float) -> bool:
        """Critère de Metropolis."""
        if delta < 0:
            return True
        if self.T <= 0:
            return False
        return random.random() < math.exp(-delta / self.T)


# ─────────────────────────────────────────────────────────────
#  6.  ALGORITHME PRINCIPAL
# ─────────────────────────────────────────────────────────────

@dataclass
class SAConfig:
    """Hyper-paramètres du recuit simulé."""
    T_init: float           = 100.0
    T_min: float            = 0.01
    alpha: float            = 0.995
    max_iter: int           = 100_000
    penalty_weight: float   = 1000.0
    target_acceptance: float= 0.20
    adapt_interval: int     = 500
    reheat_patience: int    = 3000
    reheat_factor: float    = 2.0
    reaction_factor: float  = 0.15
    segment_update: int     = 200   # fréquence de mise à jour des poids opérateurs
    verbose: bool           = True
    log_interval: int       = 5000


@dataclass
class SAResult:
    best_solution: List[List[int]]
    best_cost: float
    feasible: bool
    history: List[float] = field(default_factory=list)
    operator_stats: dict  = field(default_factory=dict)
    elapsed: float        = 0.0


def simulated_annealing(initial_solution: List[List[int]],
                        inst: Instance,
                        config: SAConfig = SAConfig()) -> SAResult:
    """
    Cœur du recuit simulé pour le VRPTW.

    Paramètres
    ----------
    initial_solution : liste de routes fournie par votre coéquipier
    inst             : instance VRPTW
    config           : hyper-paramètres

    Retourne
    --------
    SAResult avec la meilleure solution trouvée
    """
    start = time.time()

    current_sol  = copy.deepcopy(initial_solution)
    current_cost = total_cost(current_sol, inst, config.penalty_weight)

    best_sol  = copy.deepcopy(current_sol)
    best_cost = current_cost

    schedule = TemperatureSchedule(
        T_init           = config.T_init,
        T_min            = config.T_min,
        alpha            = config.alpha,
        target_acceptance= config.target_acceptance,
        adapt_interval   = config.adapt_interval,
        reheat_patience  = config.reheat_patience,
        reheat_factor    = config.reheat_factor,
    )

    selector = AdaptiveOperatorSelector(
        list(OPERATORS.keys()),
        reaction_factor=config.reaction_factor
    )

    history = []
    segment_rewards = {n: [] for n in OPERATORS}

    for it in range(1, config.max_iter + 1):

        # ── Sélection et application de l'opérateur ──────────
        op_name = selector.select()
        new_sol = OPERATORS[op_name](current_sol)
        if new_sol is None:
            selector.update(op_name, 0)
            schedule.cool()
            continue

        new_cost = total_cost(new_sol, inst, config.penalty_weight)
        delta    = new_cost - current_cost

        accepted  = schedule.accept(delta)
        improved  = new_cost < best_cost

        # ── Récompense opérateur ──────────────────────────────
        if improved:
            reward = 3
        elif accepted and delta < 0:
            reward = 2
        elif accepted:
            reward = 1
        else:
            reward = 0
        selector.update(op_name, reward)

        # ── Mise à jour solution courante ─────────────────────
        if accepted:
            current_sol  = new_sol
            current_cost = new_cost

        if improved:
            best_sol  = copy.deepcopy(new_sol)
            best_cost = new_cost

        # ── Refroidissement + enregistrement ─────────────────
        schedule.record(accepted, improved)
        schedule.cool()

        # ── Réinitialisation scores opérateurs ────────────────
        if it % config.segment_update == 0:
            selector.reset_scores()

        # ── Log ───────────────────────────────────────────────
        if it % 1000 == 0:
            history.append(best_cost)
        if config.verbose and it % config.log_interval == 0:
            feas = "✓" if is_feasible(best_sol, inst) else "✗"
            print(f"  Iter {it:>7} | T={schedule.T:8.4f} | "
                  f"Best={best_cost:10.2f} {feas} | "
                  f"Cur={current_cost:10.2f}")

    elapsed = time.time() - start

    # Statistiques finales des opérateurs
    op_stats = {
        n: {"weight": round(selector.weights[n], 4)}
        for n in selector.names
    }

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"  Terminé en {elapsed:.1f}s")
        print(f"  Meilleur coût : {best_cost:.2f}")
        print(f"  Faisable      : {is_feasible(best_sol, inst)}")
        print(f"  Nb routes     : {len(best_sol)}")
        print(f"{'='*60}")

    return SAResult(
        best_solution  = best_sol,
        best_cost      = best_cost,
        feasible       = is_feasible(best_sol, inst),
        history        = history,
        operator_stats = op_stats,
        elapsed        = elapsed,
    )


# ─────────────────────────────────────────────────────────────
#  7.  UTILITAIRES : LECTURE SOLOMON + INITIALISATION T
# ─────────────────────────────────────────────────────────────

def read_solomon(filepath: str) -> Instance:
    """
    Lit un fichier au format Solomon standard.
    Ligne 0-4 : entête
    Ligne 5   : VEHICLE  NUMBER  CAPACITY
    Ligne 6   : vide ou entête colonnes
    Ligne 7+  : CUST  XCOORD  YCOORD  DEMAND  READY_TIME  DUE_DATE  SERVICE_TIME
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Chercher la ligne véhicule
    num_vehicles, capacity = None, None
    customers_raw = []
    reading_customers = False

    for line in lines:
        tokens = line.split()
        if tokens[0].upper() == 'VEHICLE':
            continue
        if num_vehicles is None and len(tokens) == 2:
            try:
                num_vehicles = int(tokens[0])
                capacity     = float(tokens[1])
                continue
            except ValueError:
                pass
        try:
            vals = list(map(float, tokens))
            if len(vals) == 7:
                customers_raw.append(vals)
        except ValueError:
            pass

    depot_vals = customers_raw[0]
    depot = Customer(
        id=0, x=depot_vals[1], y=depot_vals[2], demand=0,
        ready_time=depot_vals[4], due_date=depot_vals[5],
        service_time=depot_vals[6]
    )
    customers = [
        Customer(
            id=int(v[0]), x=v[1], y=v[2], demand=v[3],
            ready_time=v[4], due_date=v[5], service_time=v[6]
        )
        for v in customers_raw[1:]
    ]
    return Instance(depot=depot, customers=customers,
                    vehicle_capacity=capacity,
                    num_vehicles=num_vehicles or 25)


def estimate_initial_temperature(solution: List[List[int]],
                                 inst: Instance,
                                 target_accept: float = 0.8,
                                 n_samples: int = 200) -> float:
    """
    Calibre T_init pour que ~`target_accept`% des mouvements
    détériorants soient acceptés au départ.
    Méthode : échantillonnage de deltas positifs.
    """
    deltas = []
    ops = list(OPERATORS.values())
    current = solution
    for _ in range(n_samples):
        op = random.choice(ops)
        new = op(current)
        if new is None:
            continue
        delta = total_cost(new, inst) - total_cost(current, inst)
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 100.0

    avg_delta = sum(deltas) / len(deltas)
    if avg_delta == 0:
        return 100.0
    # T tel que exp(-avg_delta / T) = target_accept
    T = -avg_delta / math.log(target_accept)
    return max(T, 1.0)


# ─────────────────────────────────────────────────────────────
#  8.  POINT D'ENTRÉE (INTERFACE AVEC VOTRE COÉQUIPIER)
# ─────────────────────────────────────────────────────────────

def run_sa_vrptw(initial_solution: List[List[int]],
                 instance_file: str,
                 config: Optional[SAConfig] = None) -> SAResult:
    """
    Fonction principale à appeler depuis votre script principal.

    Paramètres
    ----------
    initial_solution : solution initiale (format [[0,...,0], ...])
    instance_file    : chemin vers le fichier Solomon (.txt)
    config           : configuration SA (None = défauts)

    Exemple d'utilisation
    ---------------------
    from sa_vrptw import run_sa_vrptw, SAConfig

    initial = [[0, 57, 55, 54, 53, 56, 58, 60, 68, 0], ...]
    result  = run_sa_vrptw(initial, "C101.txt")
    print(result.best_cost, result.feasible)
    """
    inst = read_solomon(instance_file)

    if config is None:
        config = SAConfig()

    # Calibration automatique de T_init
    T_auto = estimate_initial_temperature(
        initial_solution, inst, target_accept=0.8
    )
    if config.T_init == SAConfig().T_init:   # valeur par défaut → remplacer
        config.T_init = T_auto
        if config.verbose:
            print(f"  T_init calibré automatiquement : {T_auto:.2f}")

    return simulated_annealing(initial_solution, inst, config)


# ─────────────────────────────────────────────────────────────
#  9.  DÉMONSTRATION AVEC DONNÉES SYNTHÉTIQUES
# ─────────────────────────────────────────────────────────────

def make_demo_instance(n: int = 100, seed: int = 42) -> Instance:
    """Crée une instance aléatoire pour tester sans fichier."""
    random.seed(seed)
    depot = Customer(0, 50, 50, 0, 0, 1000, 0)
    customers = []
    for i in range(1, n + 1):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        demand = random.uniform(5, 30)
        ready  = random.uniform(0, 400)
        due    = ready + random.uniform(50, 200)
        serv   = random.uniform(5, 15)
        customers.append(Customer(i, x, y, demand, ready, due, serv))
    return Instance(depot=depot, customers=customers,
                    vehicle_capacity=200, num_vehicles=25)


if __name__ == "__main__":
    # ── Exemple avec la solution initiale de votre coéquipier ──
    initial_solution = [[0, 57, 55, 54, 53, 56, 58, 60, 80, 0], [0, 98, 96, 95, 94, 92, 93, 97, 2, 1, 0], [0, 5, 17, 18, 19, 15, 16, 12, 4, 91, 69, 49, 0], [0, 13, 33, 31, 37, 38, 39, 36, 34, 50, 0], [0, 87, 78, 76, 71, 70, 73, 77, 68, 66, 52, 47, 0], [0, 90, 81, 83, 82, 84, 85, 88, 79, 22, 75, 0], [0, 20, 32, 35, 29, 28, 14, 23, 21, 0], [0, 43, 41, 62, 40, 44, 46, 45, 48, 51, 0], [0, 67, 63, 86, 74, 72, 61, 64, 89, 0], [0, 42, 65, 25, 27, 11, 9, 6, 100, 99, 0], [0, 24, 3, 7, 8, 10, 30, 26, 59, 0]]

    # Instance de démonstration (à remplacer par read_solomon("C101.txt"))
    inst = make_demo_instance(n=100)

    cfg = SAConfig(
        T_init           = 150.0,
        T_min            = 0.01,
        alpha            = 0.995,
        max_iter         = 50_000,
        penalty_weight   = 1000.0,
        target_acceptance= 0.20,
        adapt_interval   = 500,
        reheat_patience  = 2000,
        reheat_factor    = 2.0,
        reaction_factor  = 0.15,
        verbose          = True,
        log_interval     = 10_000,
    )

    print("=== Recuit Simulé – VRPTW ===")
    print(f"Solution initiale : {len(initial_solution)} routes")
    init_cost = total_cost(initial_solution, inst)
    print(f"Coût initial      : {init_cost:.2f}\n")

    result = simulated_annealing(initial_solution, inst, cfg)

    print("\n=== Meilleure solution ===")
    for i, route in enumerate(result.best_solution):
        d, p, feas = evaluate_route(route, inst)
        print(f"  Route {i+1:2d}: {route}  dist={d:.1f}  pen={p:.1f}  {'OK' if feas else 'INFEASIBLE'}")

    print(f"\nPoids finaux des opérateurs :")
    for op, stats in result.operator_stats.items():
        print(f"  {op:12s} : {stats['weight']:.4f}")
