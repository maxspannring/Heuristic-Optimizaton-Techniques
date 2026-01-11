import random
import time
import csv
from pathlib import Path
import os, sys

# -------------------------
# Imports from your project
# -------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from instance import Instance
from solution import Solution
from alns import *
from misc import *

# -------------------------
# Experiment configuration
# -------------------------

SIZES = [50, 100, 200, 500, 1000, 2000]
INSTANCES_PER_SIZE = 3
RUNS_PER_INSTANCE = 3
ITERS = 100
SEED_BASE = 1090

BASE_INSTANCE_DIR = Path("../instances")  # root containing 50/, 100/, ...
OUTPUT_FILE = "scaling_results.csv"

ALNS_PARAMS = dict(
    T0=10,
    cooling=0.975,
    destroy_fraction=7,
    rho=0.25,
    reward_best=7,
    reward_accept=0,
    reward_reject=-2,
    regret_k=3,
    max_gap=7
)

# -------------------------
# Helpers
# -------------------------

def sample_test_instances(n):
    test_dir = BASE_INSTANCE_DIR / str(n) / "test"

    # --- diagnostics ---
    #print(f"Checking path: {test_dir.resolve()}")
    #print(f"Exists: {test_dir.exists()}")
    #print(f"Is directory: {test_dir.is_dir()}")

    if not test_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {test_dir}")

    instances = list(test_dir.glob("*.txt"))
    print(f"Found {len(instances)} instance files")

    if len(instances) < INSTANCES_PER_SIZE:
        raise RuntimeError(
            f"Not enough test instances for n={n} "
            f"(found {len(instances)})"
        )

    return random.sample(instances, INSTANCES_PER_SIZE)


def run_single_alns(instance_path, seed):
    random.seed(seed)

    instance = Instance(str(instance_path))
    initial = Solution(instance)

    routes_array = nearest_neighbor_heuristic(instance)
    initial.load_from_arrays(routes_array)

    initial_cost = initial.total_cost

    params = ALNSParams(**ALNS_PARAMS)

    start = time.time()
    best = alns(
        instance,
        initial,
        params,
        iters=ITERS,
        log_file="/dev/null"
    )
    runtime = time.time() - start

    final_cost = best.total_cost

    return runtime, initial_cost, final_cost


# -------------------------
# Main experiment
# -------------------------
def main():
    random.seed(SEED_BASE)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "instance_file",
            "run_id",
            "seed",
            "runtime_sec",
            "initial_cost",
            "final_cost",
            "relative_improvement"
        ])

        for n in SIZES:
            instances = sample_test_instances(n)

            for instance_path in instances:
                for run in range(RUNS_PER_INSTANCE):
                    seed = SEED_BASE + run

                    runtime, initial_cost, final_cost = run_single_alns(
                        instance_path,
                        seed
                    )

                    improvement = (initial_cost - final_cost) / initial_cost

                    writer.writerow([
                        n,
                        instance_path.name,
                        run,
                        seed,
                        round(runtime, 3),
                        round(initial_cost, 3),
                        round(final_cost, 3),
                        round(improvement, 6)
                    ])

                    print(
                        f"[n={n}] {instance_path.name} | "
                        f"run {run} | "
                        f"{runtime:.1f}s | "
                        f"impr={improvement:.3f}"
                    )


if __name__ == "__main__":
    main()