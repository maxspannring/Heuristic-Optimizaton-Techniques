#!/usr/bin/env python3
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import time
import signal
from src.instance import Instance
from src.solution import Solution
from src.alns import alns, ALNSParams
from src.misc import *
import random
import numpy as np

TIME_LIMIT = 150          # 5 minutes
PENALTY = 5e3             # large objective

def timeout_handler(signum, frame):
    print(f"{PENALTY} {300}")
    sys.exit(0)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIME_LIMIT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)

    parser.add_argument("--T0", type=float)
    parser.add_argument("--cooling", type=float)
    parser.add_argument("--destroy_fraction", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--reward_best", type=float)
    parser.add_argument("--reward_accept", type=float)
    parser.add_argument("--reward_reject", type=float)
    parser.add_argument("--max_gap", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)

    args = parser.parse_args()

    try:
        #random.seed(args.seed)
        #np.random.seed(args.seed)
        #print(args.instance)
        instance = Instance(args.instance)
        #print("instance created")
        initial = Solution(instance)  # initializig initial solution
        routes_array = nearest_neighbor_heuristic(instance)  # since the NN-constr.heur. is imported from the 1st assignment, it does not natively include the new data structures yet and returns arrays
        initial.load_from_arrays(routes_array)
        #print("initial solution created")
        params = ALNSParams(
            T0=args.T0,
            cooling=args.cooling,
            destroy_fraction=args.destroy_fraction,
            rho=args.rho,
            reward_best=args.reward_best,
            reward_accept=args.reward_accept,
            reward_reject=args.reward_reject,
            regret_k=3,
            max_gap=args.max_gap
        )
        start = time.perf_counter()
        best = alns(
            instance,
            initial,
            params,
            iters=args.iters,
            log_file="/dev/null"
        )
        elapsed = time.perf_counter() - start

        print(f"{best.total_cost} {elapsed}")

    except Exception:
        print(f"{PENALTY} {300}")

if __name__ == "__main__":
    main()
