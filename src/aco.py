import numpy as np
import random
import copy
from src.solution import Solution
from src.misc import a, capacity_feasible

class FastUltraACO:
    def __init__(self, instance, alpha=1.0, beta=5.0, rho=0.05, m_ants=20, cand_size=20):
        self.instance = instance
        self.alpha, self.beta, self.rho = alpha, beta, rho
        self.m_ants = m_ants
        self.n = instance.n
        self.cand_size = cand_size

        # MMAS Initialization
        self.t_max = 1.0 / (rho * 3000)
        self.t_min = self.t_max / (self.n * 2)
        self.pheromones = np.full((self.n + 1, self.n + 1), self.t_max)
        self.stagnation_counter = 0

    def local_search_2opt(self, route_nodes):
        """Standard 2-opt refinement with feasibility checks."""
        if len(route_nodes) < 4: return route_nodes
        nodes = list(route_nodes)
        improved = True
        while improved:
            improved = False
            for i in range(len(nodes) - 1):
                for j in range(i + 2, len(nodes)):
                    new_nodes = nodes[:i] + nodes[i:j][::-1] + nodes[j:]
                    # Check distance first (fast), then feasibility (slow)
                    if self.get_path_dist(new_nodes) < self.get_path_dist(nodes):
                        if capacity_feasible(self.instance, new_nodes):
                            nodes = new_nodes
                            improved = True
            if not improved: break
        return nodes

    def get_path_dist(self, nodes):
        d = 0
        curr = self.instance.depot
        for n in nodes:
            loc = self.instance.requests[n-1]["pick_up"] if n <= self.n else self.instance.requests[n-self.n-1]["drop_off"]
            d += a(curr, loc)
            curr = loc
        return d + a(curr, self.instance.depot)

    def construct_solution(self):
        """Pure construction phase (no local search here to save time)."""
        sol = Solution(self.instance)
        routes_nodes = [[] for _ in range(self.instance.n_k)]
        unserved = set(range(1, self.n + 1))
        curr_locs = [self.instance.depot] * self.instance.n_k
        curr_nodes = [0] * self.instance.n_k
        v_dist = [0.0] * self.instance.n_k
        capacities = [0] * self.instance.n_k
        carrying = [[] for _ in range(self.instance.n_k)]
        served_count = 0

        while served_count < self.instance.gamma or any(carrying):
            possible_moves = []
            avg_dist = sum(v_dist) / self.instance.n_k

            for k in range(self.instance.n_k):
                if served_count < self.instance.gamma and unserved:
                    cands = sorted(list(unserved),
                                   key=lambda r: a(curr_locs[k], self.instance.requests[r-1]["pick_up"]))[:self.cand_size]
                    for r in cands:
                        if capacities[k] + self.instance.requests[r-1]["demand"] <= self.instance.C:
                            possible_moves.append(('pickup', k, r))
                for r in carrying[k]:
                    possible_moves.append(('dropoff', k, r))

            if not possible_moves: break
            vals = []
            for action, v_idx, req_id in possible_moves:
                target = self.instance.requests[req_id-1]["pick_up"] if action == 'pickup' else self.instance.requests[req_id-1]["drop_off"]
                tau = self.pheromones[curr_nodes[v_idx]][req_id] if action == 'pickup' else self.t_max
                penalty = 1.0 / (1.0 + max(0, v_dist[v_idx] - avg_dist)**2.5)
                eta = (1.0 / max(a(curr_locs[v_idx], target), 1)) * penalty
                vals.append((tau**self.alpha) * (eta**self.beta))

            choice = random.choices(possible_moves, weights=vals, k=1)[0]
            action, v_idx, req_id = choice
            target_loc = self.instance.requests[req_id-1]["pick_up"] if action == 'pickup' else self.instance.requests[req_id-1]["drop_off"]
            v_dist[v_idx] += a(curr_locs[v_idx], target_loc)

            if action == 'pickup':
                routes_nodes[v_idx].append(req_id); unserved.remove(req_id); carrying[v_idx].append(req_id)
                capacities[v_idx] += self.instance.requests[req_id-1]["demand"]
                curr_nodes[v_idx], curr_locs[v_idx] = req_id, target_loc
            else:
                routes_nodes[v_idx].append(req_id + self.n); carrying[v_idx].remove(req_id)
                capacities[v_idx] -= self.instance.requests[req_id-1]["demand"]
                curr_locs[v_idx] = target_loc; served_count += 1

        sol.load_from_arrays(routes_nodes)
        return sol

    def run(self, iterations=100):
        global_best = None
        for i in range(iterations):
            # 1. Ants construct raw solutions
            ants = [self.construct_solution() for _ in range(self.m_ants)]
            valid = [s for s in ants if s.is_feasible()]
            if not valid: continue

            # 2. Find the best ant of THIS iteration
            it_best = min(valid, key=lambda s: s.total_cost)

            # 3. Apply Local Search ONLY to the iteration best (Saves massive time)
            refined_routes = []
            for route in it_best.routes:
                refined_routes.append(self.local_search_2opt(route.nodes))
            it_best.load_from_arrays(refined_routes)

            # 4. Update Global Best
            if global_best is None or it_best.total_cost < global_best.total_cost:
                global_best = it_best.copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # 5. MMAS Pheromone Update
            self.pheromones = np.clip(self.pheromones * (1 - self.rho), self.t_min, self.t_max)
            reward = 3000.0 / (global_best.total_cost + 1)
            for route in global_best.routes:
                prev = 0
                for node in route.nodes:
                    if node <= self.n:
                        self.pheromones[prev][node] = min(self.t_max, self.pheromones[prev][node] + reward)
                        prev = node

            if self.stagnation_counter > 20:
                self.pheromones = (self.pheromones + self.t_max) / 2
                self.stagnation_counter = 0

            #if i % 10 == 0:
                #print(f"Iter {i}: Cost {global_best.total_cost:.2f}, Fairness {global_best.fairness:.3f}")
        return global_best