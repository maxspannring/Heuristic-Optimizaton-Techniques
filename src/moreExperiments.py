#!/usr/bin/env python3
import os
import sys
import time
import csv
from pathlib import Path
import random
import numpy as np

# -------------------------------------------------
# Project setup
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.instance import Instance
from src.solution import Solution
from src.alns import alns, ALNSParams
from src.misc import nearest_neighbor_heuristic
from src.beam_search import beam_search           # <-- check import path
from src.aco import FastUltraACO                  # <-- check import path

# -------------------------------------------------
# Experiment configuration
# -------------------------------------------------
INSTANCE_DIR = Path("../instances/100/test")
OUTPUT_FILE = "test_results_n100.csv"

SEED = 1090          # <-- you may want multiple seeds later
random.seed(SEED)
np.random.seed(SEED)

# ---------------- ALNS parameters ----------------
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
ALNS_ITERS = 2000     # <-- adjust if runtime is too high

# ---------------- ACO parameters -----------------
ACO_PARAMS = dict(
    alpha=2.0,
    beta=6.0,
    rho=0.1,
    m_ants=20,
    cand_size=30
)
ACO_ITERS = 50        # <-- increase to 100 if runtime allows

# ---------------- Beam Search --------------------
BEAM_WIDTH = 200      # <-- main tuning knob for beam search

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def run_alns(instance):
    """Run ALNS and return (runtime, objective)."""
    s0 = Solution(instance)

    # Initial solution via NN heuristic
    routes_array = nearest_neighbor_heuristic(instance)
    s0.load_from_arrays(routes_array)

    params = ALNSParams(**ALNS_PARAMS)

    start = time.perf_counter()
    best = alns(instance, s0, params, iters=ALNS_ITERS, log_file="/dev/null")
    runtime = time.perf_counter() - start

    return runtime, best.total_cost


def run_aco(instance):
    """Run ACO and return (runtime, objective)."""
    aco = FastUltraACO(instance, **ACO_PARAMS)

    start = time.perf_counter()
    best = aco.run(iterations=ACO_ITERS)
    runtime = time.perf_counter() - start

    return runtime, best.total_cost


def run_beam_search(instance):
    """Run Beam Search and return (runtime, objective)."""
    start = time.perf_counter()
    routes = beam_search(instance, BEAM_WIDTH)
    runtime = time.perf_counter() - start

    # Convert routes (array-of-arrays) into Solution
    sol = Solution(instance)
    sol.load_from_arrays(routes)

    return runtime, sol.total_cost


# -------------------------------------------------
# Main experiment loop
# -------------------------------------------------
def main():
    instance_files = sorted(INSTANCE_DIR.glob("*.txt"))

    if not instance_files:
        raise RuntimeError("No test instances found.")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance",
            "algorithm",
            "runtime_sec",
            "objective"
        ])

        for inst_path in instance_files:
            print(f"\n=== Instance: {inst_path.name} ===")
            instance = Instance(str(inst_path))

            # -------- ALNS --------
            try:
                rt, obj = run_alns(instance)
                writer.writerow([inst_path.name, "ALNS", round(rt, 3), round(obj, 3)])
                print(f"ALNS       | {rt:7.1f}s | obj={obj:.2f}")
            except Exception as e:
                print("ALNS failed:", e)
                writer.writerow([inst_path.name, "ALNS", None, None])

            # -------- ACO --------
            try:
                rt, obj = run_aco(instance)
                writer.writerow([inst_path.name, "ACO", round(rt, 3), round(obj, 3)])
                print(f"ACO        | {rt:7.1f}s | obj={obj:.2f}")
            except Exception as e:
                print("ACO failed:", e)
                writer.writerow([inst_path.name, "ACO", None, None])

            # -------- Beam Search --------
            try:
                rt, obj = run_beam_search(instance)
                writer.writerow([inst_path.name, "BeamSearch", round(rt, 3), round(obj, 3)])
                print(f"BeamSearch | {rt:7.1f}s | obj={obj:.2f}")
            except Exception as e:
                print("Beam Search failed:", e)
                writer.writerow([inst_path.name, "BeamSearch", None, None])


if __name__ == "__main__":
    main()
