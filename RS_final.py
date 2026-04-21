"""
=============================================================
  Recuit Simulé (Simulated Annealing) pour le VRPTW
=============================================================
"""

import math
import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from cooling_strategies.adaptive import TemperatureSchedule


# ─────────────────────────────────────────────────────────────
#  1.  STRUCTURES DE DONNÉES
# ─────────────────────────────────────────────────────────────

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class Instance:
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: float
    num_vehicles: int

    def node(self, idx: int) -> Customer:
        if idx == 0:
            return self.depot
        return self.customers[idx - 1]

    def distance(self, i: int, j: int) -> float:
        a, b = self.node(i), self.node(j)
        return math.hypot(a.x - b.x, a.y - b.y)


# ─────────────────────────────────────────────────────────────
#  2.  VÉRIFICATION STRICTE DE FAISABILITÉ
# ─────────────────────────────────────────────────────────────

def check_faisabilite(solution: List[List[int]], inst: Instance) -> bool:
    """
    Vérifie STRICTEMENT toutes les contraintes du VRPTW :
    - Chaque client visité exactement une seule fois
    - Capacité du véhicule non dépassée
    - Fenêtre de temps de chaque client respectée (ready_time et due_date)
    - Temps de service de chaque client respecté
    - Fenêtre de temps du dépôt respectée (retour avant l_depot)
    - Chaque route commence et finit au dépôt (structure [0,...,0])
    """
    n = len(inst.customers)
    marquage_clients = [False] * (n + 1)  # index 1..n

    for route in solution:
        # Vérification structure : commence et finit au dépôt
        if route[0] != 0 or route[-1] != 0:
            return False

        charge = 0.0
        temps = inst.depot.ready_time   # e_depot
        prev = 0                         # on part du dépôt

        for i in route[1:-1]:            # clients hors dépôts
            # Contrainte 1 : unicité — client déjà visité ?
            if marquage_clients[i]:
                return False
            marquage_clients[i] = True

            # Contrainte 2 : capacité
            charge += inst.node(i).demand
            if charge > inst.vehicle_capacity:
                return False

            # Contrainte 3 : fenêtre de temps client
            temps += inst.distance(prev, i)
            cust = inst.node(i)
            if temps < cust.ready_time:
                temps = cust.ready_time          # attente autorisée
            if temps > cust.due_date:
                return False                     # violation → infaisable

            # Contrainte 4 : temps de service
            temps += cust.service_time
            prev = i

        # Contrainte 5 : retour au dépôt — fenêtre de temps dépôt
        temps += inst.distance(prev, 0)
        if temps > inst.depot.due_date:          # l_depot
            return False

    # Contrainte 6 : tous les clients ont été visités exactement une fois
    for i in range(1, n + 1):
        if not marquage_clients[i]:
            return False

    return True


# ─────────────────────────────────────────────────────────────
#  3.  ÉVALUATION D'UNE SOLUTION (pour le coût SA)
# ─────────────────────────────────────────────────────────────

def evaluate_route(route: List[int], inst: Instance
                   ) -> Tuple[float, float, bool]:
    """
    Évalue une route pour le recuit simulé.
    Retourne (distance_totale, pénalité_contraintes, faisable).
    La pénalité est utilisée uniquement pour guider le SA —
    la faisabilité stricte est vérifiée par check_faisabilite.
    """
    dist = 0.0
    load = 0.0
    time_now = inst.depot.ready_time
    penalty = 0.0
    feasible = True
    prev = 0

    for k in range(1, len(route) - 1):
        j = route[k]
        d = inst.distance(prev, j)
        dist += d
        time_now += d

        cust = inst.node(j)
        load += cust.demand

        if time_now < cust.ready_time:
            time_now = cust.ready_time
        if time_now > cust.due_date:
            penalty += (time_now - cust.due_date)
            feasible = False
        time_now += cust.service_time
        prev = j

    # Distance retour dépôt
    d_retour = inst.distance(prev, 0)
    dist += d_retour
    time_now += d_retour

    # Pénalité capacité
    if load > inst.vehicle_capacity:
        penalty += (load - inst.vehicle_capacity) * 1000
        feasible = False

    # Pénalité fenêtre de temps dépôt
    if time_now > inst.depot.due_date:
        penalty += (time_now - inst.depot.due_date)
        feasible = False

    return dist, penalty, feasible


def total_cost(solution: List[List[int]], inst: Instance,
               penalty_weight: float = 1000.0,
                 vehicule_weight: float=10000.0) -> float:
    total = 0.0
    for route in solution:
        d, p, _ = evaluate_route(route, inst)
        total += d + penalty_weight * p
    return total + vehicule_weight * len(solution)


def is_feasible(solution: List[List[int]], inst: Instance) -> bool:
    """Utilise check_faisabilite pour une vérification stricte."""
    return check_faisabilite(solution, inst)


# ─────────────────────────────────────────────────────────────
#  4.  OPÉRATEURS DE VOISINAGE
# ─────────────────────────────────────────────────────────────

def _clients_uniques(solution: List[List[int]], inst: Instance) -> bool:
    """
    Vérifie rapidement qu'aucun client n'est dupliqué ou manquant
    après une transformation — utilisé dans chaque opérateur.
    """
    n = len(inst.customers)
    seen = set()
    for route in solution:
        for c in route:
            if c == 0:
                continue
            if c in seen:
                return False
            seen.add(c)
    return len(seen) == n


# ---------- 4a. Relocate ----------
def neighbor_relocate(solution: List[List[int]],
                      inst: Instance) -> Optional[List[List[int]]]:
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 2]
    if not routes_with_clients:
        return None

    r1_idx = random.choice(routes_with_clients)
    route1 = sol[r1_idx]
    pos = random.randint(1, len(route1) - 2)
    client = route1[pos]

    new_r1 = route1[:pos] + route1[pos + 1:]
    if len(new_r1) == 2:
        sol.pop(r1_idx)
    else:
        sol[r1_idx] = new_r1

    r2_candidates = list(range(len(sol)))
    if not r2_candidates:
        return None

    r2_idx = random.choice(r2_candidates)
    route2 = sol[r2_idx]
    ins_pos = random.randint(1, len(route2) - 1)
    sol[r2_idx] = route2[:ins_pos] + [client] + route2[ins_pos:]

    if not _clients_uniques(sol, inst):
        return None
    return sol


# ---------- 4b. Swap ----------
def neighbor_swap(solution: List[List[int]],
                  inst: Instance) -> Optional[List[List[int]]]:
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 2]
    if len(routes_with_clients) < 2:
        return None

    r1_idx, r2_idx = random.sample(routes_with_clients, 2)
    r1, r2 = sol[r1_idx], sol[r2_idx]

    p1 = random.randint(1, len(r1) - 2)
    p2 = random.randint(1, len(r2) - 2)
    r1[p1], r2[p2] = r2[p2], r1[p1]

    if not _clients_uniques(sol, inst):
        return None
    return sol


# ---------- 4c. 2-opt ----------
def neighbor_2opt(solution: List[List[int]],
                  inst: Instance) -> Optional[List[List[int]]]:
    sol = copy.deepcopy(solution)
    routes_with_clients = [i for i, r in enumerate(sol) if len(r) > 3]
    if not routes_with_clients:
        return None

    r_idx = random.choice(routes_with_clients)
    route = sol[r_idx]
    n = len(route) - 2
    if n < 2:
        return None

    i = random.randint(1, len(route) - 3)
    j = random.randint(i + 1, len(route) - 2)
    route[i:j + 1] = route[i:j + 1][::-1]
    sol[r_idx] = route

    if not _clients_uniques(sol, inst):
        return None
    return sol


# ---------- 4d. Or-opt ----------
def neighbor_or_opt(solution: List[List[int]],
                    inst: Instance,
                    seg_len: int = 2) -> Optional[List[List[int]]]:
    sol = copy.deepcopy(solution)
    candidates = [i for i, r in enumerate(sol) if len(r) - 2 >= seg_len]
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
    else:
        sol[r1_idx] = new_r1

    r2_candidates = list(range(len(sol)))
    if not r2_candidates:
        return None

    r2_idx = random.choice(r2_candidates)
    route2 = sol[r2_idx]
    ins_pos = random.randint(1, len(route2) - 1)
    sol[r2_idx] = route2[:ins_pos] + segment + route2[ins_pos:]

    if not _clients_uniques(sol, inst):
        return None
    return sol


# ---------- 4e. Cross ----------
def neighbor_cross(solution: List[List[int]],
                   inst: Instance) -> Optional[List[List[int]]]:
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

    if not _clients_uniques(sol, inst):
        return None
    return sol


OPERATORS = {
    "relocate": neighbor_relocate,
    "swap":     neighbor_swap,
    "2opt":     neighbor_2opt,
    "or_opt_1": lambda s, inst: neighbor_or_opt(s, inst, 1),
    "or_opt_2": lambda s, inst: neighbor_or_opt(s, inst, 2),
    "or_opt_3": lambda s, inst: neighbor_or_opt(s, inst, 3),
    "cross":    neighbor_cross,
}


# ─────────────────────────────────────────────────────────────
#  5.  SÉLECTION ADAPTATIVE DES OPÉRATEURS
# ─────────────────────────────────────────────────────────────

class AdaptiveOperatorSelector:
    def __init__(self, operator_names: List[str],
                 reaction_factor: float = 0.1):
        self.names = operator_names
        self.weights = {n: 1.0 for n in operator_names}
        self.scores  = {n: 0.0 for n in operator_names}
        self.uses    = {n: 0   for n in operator_names}
        self.r = reaction_factor

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
        self.uses[name] += 1
        self.scores[name] += reward
        if self.uses[name] > 0:
            avg = self.scores[name] / self.uses[name]
            self.weights[name] = (1 - self.r) * self.weights[name] + self.r * avg

    def reset_scores(self):
        self.scores = {n: 0.0 for n in self.names}
        self.uses   = {n: 0   for n in self.names}


# ─────────────────────────────────────────────────────────────
#  6.  SCHÉMA DE REFROIDISSEMENT ADAPTATIF
# ─────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────
#  7.  ALGORITHME PRINCIPAL
# ─────────────────────────────────────────────────────────────

@dataclass
class SAConfig:
    T_init: float            = 100.0
    T_min: float             = 0.01
    alpha: float             = 0.995
    max_iter: int            = 100_000
    penalty_weight: float    = 1000.0
    target_acceptance: float = 0.20
    adapt_interval: int      = 500
    reheat_patience: int     = 3000
    reheat_factor: float     = 2.0
    reaction_factor: float   = 0.15
    segment_update: int      = 200
    verbose: bool            = True
    log_interval: int        = 5000


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
                        config: SAConfig = SAConfig(),
                        custom_schedule = None) -> SAResult:

    # Vérification stricte de la solution initiale
    if not check_faisabilite(initial_solution, inst):
        if config.verbose:
            print("    Solution initiale infaisable — poursuite avec pénalités")

    start = time.time()

    current_sol  = copy.deepcopy(initial_solution)
    current_cost = total_cost(current_sol, inst, config.penalty_weight)

    best_sol  = copy.deepcopy(current_sol)
    best_cost = current_cost

    if custom_schedule is not None:
        # On utilise la stratégie injectée (Logarithmique, Paliers, etc.)
        schedule = custom_schedule
    else:
        # Fallback de sécurité : si on ne passe rien, on garde la stratégie par défaut de ton ami
        schedule = TemperatureSchedule(
            T_init            = config.T_init,
            T_min             = config.T_min,
            alpha             = config.alpha,
            target_acceptance = config.target_acceptance,
            adapt_interval    = config.adapt_interval,
            reheat_patience   = config.reheat_patience,
            reheat_factor     = config.reheat_factor,
        )

    selector = AdaptiveOperatorSelector(
        list(OPERATORS.keys()),
        reaction_factor=config.reaction_factor
    )

    history = []

    for it in range(1, config.max_iter + 1):

        op_name = selector.select()
        new_sol = OPERATORS[op_name](current_sol, inst)   # inst passé en paramètre
        if new_sol is None:
            selector.update(op_name, 0)
            schedule.cool()
            continue

        new_cost = total_cost(new_sol, inst, config.penalty_weight)
        delta    = new_cost - current_cost

        accepted = schedule.accept(delta)
        improved = new_cost < best_cost

        if improved:
            reward = 3
        elif accepted and delta < 0:
            reward = 2
        elif accepted:
            reward = 1
        else:
            reward = 0
        selector.update(op_name, reward)

        if accepted:
            current_sol  = new_sol
            current_cost = new_cost

        if improved:
            best_sol  = copy.deepcopy(new_sol)
            best_cost = new_cost

        schedule.record(accepted, improved)
        schedule.cool()

        if it % config.segment_update == 0:
            selector.reset_scores()

        if it % 1000 == 0:
            history.append(best_cost)

        if config.verbose and it % config.log_interval == 0:
            # Utilise check_faisabilite pour le log — vérification stricte
            feas = "✓" if check_faisabilite(best_sol, inst) else "✗"
            print(f"  Iter {it:>7} | T={schedule.T:8.4f} | "
                  f"Best={best_cost:10.2f} {feas} | "
                  f"Cur={current_cost:10.2f}")

    elapsed = time.time() - start

    op_stats = {
        n: {"weight": round(selector.weights[n], 4)}
        for n in selector.names
    }

    # Vérification finale stricte avec check_faisabilite
    solution_feasible = check_faisabilite(best_sol, inst)

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"  Terminé en {elapsed:.1f}s")
        print(f"  Meilleur coût : {best_cost:.2f}")
        print(f"  Faisable      : {solution_feasible}")
        print(f"  Nb routes     : {len(best_sol)}")
        print(f"{'='*60}")

    return SAResult(
        best_solution  = best_sol,
        best_cost      = best_cost,
        feasible       = solution_feasible,
        history        = history,
        operator_stats = op_stats,
        elapsed        = elapsed,
    )


# ─────────────────────────────────────────────────────────────
#  8.  UTILITAIRES
# ─────────────────────────────────────────────────────────────

def read_solomon(filepath: str) -> Instance:
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    num_vehicles, capacity = None, None
    customers_raw = []

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
    deltas = []
    ops = list(OPERATORS.values())
    current = solution
    for _ in range(n_samples):
        op = random.choice(ops)
        new = op(current, inst)
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
    T = -avg_delta / math.log(target_accept)
    return max(T, 1.0)


def run_sa_vrptw(initial_solution: List[List[int]],
                 instance_file: str,
                 config: Optional[SAConfig] = None) -> SAResult:
    inst = read_solomon(instance_file)

    if config is None:
        config = SAConfig()

    T_auto = estimate_initial_temperature(initial_solution, inst, target_accept=0.8)
    if config.T_init == SAConfig().T_init:
        config.T_init = T_auto
        if config.verbose:
            print(f"  T_init calibré automatiquement : {T_auto:.2f}")

    return simulated_annealing(initial_solution, inst, config)


# ─────────────────────────────────────────────────────────────
#  9.  DÉMONSTRATION
# ─────────────────────────────────────────────────────────────

def make_demo_instance(n: int = 100, seed: int = 42) -> Instance:
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
def diagnose_initial_solution(solution: List[List[int]], inst: Instance):
    """
    Prints exactly which constraints are violated in the initial solution
    and where.
    """
    print("\n=== DIAGNOSTIC SOLUTION INITIALE ===")
    n = len(inst.customers)
    marquage_clients = [False] * (n + 1)
    total_violations = 0

    for r_idx, route in enumerate(solution):

        # Check 1 : structure dépôt
        if route[0] != 0 or route[-1] != 0:
            print(f"  [Route {r_idx+1}]  Ne commence/finit pas au dépôt : {route}")
            total_violations += 1

        charge = 0.0
        temps = inst.depot.ready_time
        prev = 0

        for i in route[1:-1]:

            # Check 2 : unicité
            if marquage_clients[i]:
                print(f"  [Route {r_idx+1}]  Client {i} visité plusieurs fois")
                total_violations += 1
            marquage_clients[i] = True

            # Check 3 : capacité
            charge += inst.node(i).demand
            if charge > inst.vehicle_capacity:
                print(f"  [Route {r_idx+1}]  Capacité dépassée après client {i} "
                      f": charge={charge:.1f} > Q={inst.vehicle_capacity:.1f}")
                total_violations += 1

            # Check 4 : fenêtre de temps client
            temps += inst.distance(prev, i)
            cust = inst.node(i)
            if temps < cust.ready_time:
                temps = cust.ready_time
            if temps > cust.due_date:
                print(f"  [Route {r_idx+1}]  Fenêtre temps client {i} violée "
                      f": arrivée={temps:.1f} > due_date={cust.due_date:.1f} "
                      f"(retard={temps - cust.due_date:.1f})")
                total_violations += 1

            temps += cust.service_time
            prev = i

        # Check 5 : retour dépôt
        temps += inst.distance(prev, 0)
        if temps > inst.depot.due_date:
            print(f"  [Route {r_idx+1}]  Retour dépôt trop tard "
                  f": temps={temps:.1f} > l_depot={inst.depot.due_date:.1f} "
                  f"(retard={temps - inst.depot.due_date:.1f})")
            total_violations += 1

    # Check 6 : clients manquants
    manquants = [i for i in range(1, n + 1) if not marquage_clients[i]]
    if manquants:
        print(f"   Clients non visités : {manquants}")
        total_violations += 1

    if total_violations == 0:
        print("   Aucune violation détectée — solution initiale faisable")
    else:
        print(f"\n  Total violations : {total_violations}")
    print("=====================================\n")

if __name__ == "__main__":
    initial_solution = [[0, 57, 55, 54, 53, 56, 58, 60, 80, 0], [0, 98, 96, 95, 94, 92, 93, 97, 2, 1, 0], [0, 5, 17, 18, 19, 15, 16, 12, 4, 91, 69, 49, 0], [0, 13, 33, 31, 37, 38, 39, 36, 34, 50, 0], [0, 87, 78, 76, 71, 70, 73, 77, 68, 66, 52, 47, 0], [0, 90, 81, 83, 82, 84, 85, 88, 79, 22, 75, 0], [0, 20, 32, 35, 29, 28, 14, 23, 21, 0], [0, 43, 41, 62, 40, 44, 46, 45, 48, 51, 0], [0, 67, 63, 86, 74, 72, 61, 64, 89, 0], [0, 42, 65, 25, 27, 11, 9, 6, 100, 99, 0], [0, 24, 3, 7, 8, 10, 30, 26, 59, 0]]

    inst = make_demo_instance(n=100)
    diagnose_initial_solution(initial_solution, inst)

    cfg = SAConfig(
        T_init            = 150.0,
        T_min             = 0.01,
        alpha             = 0.995,
        max_iter          = 50_000,
        penalty_weight    = 1000.0,
        target_acceptance = 0.20,
        adapt_interval    = 500,
        reheat_patience   = 2000,
        reheat_factor     = 2.0,
        reaction_factor   = 0.15,
        verbose           = True,
        log_interval      = 10_000,
    )

    print("=== Recuit Simulé – VRPTW ===")
    print(f"Solution initiale : {len(initial_solution)} routes")

    # Vérification stricte de la solution initiale
    print(f"Faisable initiale : {check_faisabilite(initial_solution, inst)}")
    init_cost = total_cost(initial_solution, inst)
    print(f"Coût initial      : {init_cost:.2f}\n")

    result = simulated_annealing(initial_solution, inst, cfg)

    print("\n=== Meilleure solution ===")
    for i, route in enumerate(result.best_solution):
        d, p, _ = evaluate_route(route, inst)
        print(f"  Route {i+1:2d}: {route}  dist={d:.1f}  pen={p:.1f}  {'OK' if p == 0 else 'INFEASIBLE'}")

    print(f"\nVérification stricte finale : {check_faisabilite(result.best_solution, inst)}")

    print(f"\nPoids finaux des opérateurs :")
    for op, stats in result.operator_stats.items():
        print(f"  {op:12s} : {stats['weight']:.4f}")