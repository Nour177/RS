import pandas as pd
import json
import random
from pathlib import Path

import RS_final as backend
from app2 import build_heuristic_solution
from cooling_strategies.adaptive import TemperatureSchedule
from cooling_strategies.logarithmique import LogarithmicSchedule
from cooling_strategies.par_paliers import StepSchedule


HEURISTICS = [
    "Nearest Neighbor",
    "Solomon Insertion",
    "Tour Géant",
    "Regret-K",
    "Clarke-Wright",
]

instances = ["C101.txt", "C207.txt", "R101.txt", "R208.txt", "RC101.txt","RC206.txt"]
strategies = ["adaptive", "logarithmic", "step"]

results_dir = Path("tuning_results")
results_dir.mkdir(exist_ok=True)

for heuristic in HEURISTICS:
    results = []
    print(f"\n{'='*60}")
    print(f"Traitement de l'heuristique: {heuristic}")
    print(f"{'='*60}")
    
    for inst_file in instances:
        inst = backend.read_solomon(f"Archive/{inst_file}")

        for strategy in strategies:

            base_config = {
                "max_iter": 20000,
                "penalty_weight": 1000,
                "T_min": 0.01,
            }

            if strategy == "adaptive":
                configs = [
                    {"alpha": 0.995, "target_acceptance": 0.2, "adapt_interval": 500, "reheat_factor": 2.0},
                    {"alpha": 0.99, "target_acceptance": 0.1, "adapt_interval": 300, "reheat_factor": 1.5},
                ]

            elif strategy == "logarithmic":
                configs = [
                    {"cooling_factor": 1.0},
                    {"cooling_factor": 2.0},
                ]

            elif strategy == "step":
                configs = [
                    {"alpha": 0.90, "longueur_palier": 1000},
                    {"alpha": 0.95, "longueur_palier": 500},
                ]

            for cfg in configs:

                initial_sol = build_heuristic_solution(heuristic, inst)

                init_cost = backend.total_cost(initial_sol, inst, 1000)

                T_init = backend.estimate_initial_temperature(initial_sol, inst)
                T_min = 0.01
                
                sa_config = backend.SAConfig(
                    T_init=T_init,
                    T_min=T_min,
                    max_iter=20000,
                    penalty_weight=1000
                )
                if strategy == "adaptive":
                    schedule = TemperatureSchedule(T_init, T_min, **cfg)
                elif strategy == "logarithmic":
                    schedule = LogarithmicSchedule(T_init, T_min, max_iter=20000, **cfg)
                else:
                    schedule = StepSchedule(T_init, T_min, max_iter=20000, **cfg)

                sa_result = backend.simulated_annealing(
                    initial_sol, inst, sa_config, schedule
                )

                final_cost = sa_result.best_cost

                results.append({
                    "heuristic": heuristic,
                    "instance": inst_file,
                    "strategy": strategy,
                    "params": cfg,
                    "initial_cost": init_cost,
                    "final_cost": final_cost,
                    "improvement_pct": (init_cost - final_cost) / init_cost * 100 if init_cost > 0 else 0,
                    "feasible": sa_result.feasible,
                    "elapsed": sa_result.elapsed,
                })
                
                print(f"  ✓ {inst_file} | {strategy} | Coût: {final_cost:.2f} | Amélioration: {(init_cost - final_cost) / init_cost * 100:.2f}%")
    

    heuristic_name = heuristic.replace(" ", "_").replace("-", "_").lower()
    
    df = pd.DataFrame(results)
    csv_file = results_dir / f"{heuristic_name}_results.csv"
    json_file = results_dir / f"{heuristic_name}_results.json"
    
    df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")

print(f"\n{'='*60}")
print("Tuning terminé! ✓")
print(f"{'='*60}")