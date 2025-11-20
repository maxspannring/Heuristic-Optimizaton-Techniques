import heapq
from itertools import islice
import random
import matplotlib.pyplot as plt

def build_qubo(A):
    """Build QUBO matrix for the bipartition problem."""
    n = len(A)
    Q = [[0] * n for _ in range(n)]
    total = sum(A)

    for i in range(n):
        Q[i][i] = 4 * A[i] ** 2 - 4 * A[i] * total  # Added linear term
        for j in range(i + 1, n):
            Q[i][j] = Q[j][i] = 4 * A[i] * A[j]
    return Q


def evaluate_partial(x, Q):
    """Evaluate partial assignment x."""
    xv = [(0 if b is None else b) for b in x]

    val = 0
    n = len(x)
    for i in range(n):
        for j in range(n):
            val += xv[i] * Q[i][j] * xv[j]
    return val


def beam_search_partition(A, beta=2):
    """
    Beam search for the QUBO bipartition.

    A: list of numbers
    beta: beam width
    """
    n = len(A)
    Q = build_qubo(A)

    # Root is completely unassigned: [None,...]
    beam = [[None] * n]

    for depth in range(n):
        candidates = []

        for node in beam:
            # Branch: assign 0 or 1 to x[depth]
            for bit in [0, 1]:
                new_node = node.copy()
                new_node[depth] = bit
                val = evaluate_partial(new_node, Q)
                candidates.append((val, new_node))

        # Keep only the best beta nodes
        candidates.sort(key=lambda x: x[0])
        beam = [node for (_, node) in candidates[:beta]]

    # After full assignment: pick best from final beam
    best_val, best_node = min(
        [(evaluate_partial(node, Q), node) for node in beam],
        key=lambda x: x[0]
    )

    # Translate to E and F
    E = [a for a, x in zip(A, best_node) if x == 1]
    F = [a for a, x in zip(A, best_node) if x == 0]

    return best_node, E, F, best_val


# --- Example usage ---
if __name__ == "__main__":
    random.seed(444)

    # random instance
    A = [random.randint(1, 10000) for _ in range(20)]

    beta_values = [1, 2, 3, 5, 10, 20, 40, 80, 120, 200]
    diffs = []

    for beta in beta_values:
        _, E, F, _ = beam_search_partition(A, beta)
        diff = abs(sum(E) - sum(F))
        diffs.append(diff)
        print(f"beta={beta:3d} -> diff={diff}")

    # plot
    plt.plot(beta_values, diffs, marker="o")
    plt.xlabel("Beam width β")
    plt.ylabel("Partition difference |sum(E) - sum(F)|")
    plt.title("Beam Search Quality vs Beam Width")
    plt.grid(True)
    plt.show()