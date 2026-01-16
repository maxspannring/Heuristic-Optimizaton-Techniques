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
from src.misc import *
import random
import numpy as np

# Import your ACO class (adjust the import path as needed)
# Assuming it's in src/aco.py or similar
from src.aco import FastUltraACO

TIME_LIMIT = 150  # 2.5 minutes
PENALTY = 5e3  # large objective for failures


def timeout_handler(signum, frame):
    print(f"{PENALTY} {300}")
    sys.exit(0)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIME_LIMIT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)

    # ACO parameters
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--m_ants", type=int)
    parser.add_argument("--cand_size", type=int)

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)

    args = parser.parse_args()

    try:
        # Set random seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Load instance
        instance = Instance(args.instance)

        # Initialize ACO with provided parameters
        aco = FastUltraACO(
            instance,
            alpha=args.alpha,
            beta=args.beta,
            rho=args.rho,
            m_ants=args.m_ants,
            cand_size=args.cand_size
        )

        # Run ACO
        start = time.perf_counter()
        best_sol = aco.run(iterations=args.iters)
        elapsed = time.perf_counter() - start

        # Output: cost and time
        if best_sol is not None and best_sol.is_feasible():
            print(f"{best_sol.total_cost} {elapsed}")
        else:
            print(f"{PENALTY} {elapsed}")

    except Exception as e:
        # If anything goes wrong, output penalty
        print(f"{PENALTY} {300}")
        # Optionally log the error for debugging
        # print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()