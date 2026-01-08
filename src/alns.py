import random
from src.misc import *
from src.instance import Instance
from src.route import Route
from src.solution import Solution
import csv
import time
import random
import math
from dataclasses import dataclass

# === [ DESTROY OPERATORS ] ===
def random_removal(solution, q):
    removed = set()
    all_requests = list(
        req for r in solution.routes for req in r.requests
    )
    random.shuffle(all_requests)

    for req in all_requests[:q]:
        route_idx = solution.request_to_route[req]
        del solution.request_to_route[req]
        route = solution.routes[route_idx]
        route.remove_request(req)
        solution.unserved_requests.add(req)
        removed.add(req)

    return removed


def worst_removal(solution, q):
    candidates = []

    for route in solution.routes:
        for req in route.requests:
            delta = route.marginal_cost(req)
            candidates.append((delta, req))

    candidates.sort(reverse=True)
    removed = set()

    for _, req in candidates[:q]:
        route_idx = solution.request_to_route[req]
        del solution.request_to_route[req]
        route = solution.routes[route_idx]
        route.remove_request(req)
        solution.unserved_requests.add(req)
        removed.add(req)

    return removed

def shaw_removal(solution, q):
    removed = set()
    all_requests = list(
        req for r in solution.routes for req in r.requests
    )
    seed = random.choice(all_requests)
    removed.add(seed)

    def relatedness(r1, r2):
        #print(f"request 1: {r1}, request2: {r2}")
        r1_idx = int(r1 - 1)
        r2_idx = int(r2 - 1)
        p1 = solution.instance.requests[r1_idx]["pick_up"]
        p2 = solution.instance.requests[r2_idx]["pick_up"]
        return a(p1, p2)


    while len(removed) < q:
        candidates = [
            r for r in all_requests if r not in removed
        ]
        candidates.sort(key=lambda r: relatedness(seed, r))
        removed.add(candidates[0])

    for req in removed:
        route_idx = solution.request_to_route[req]
        route = solution.routes[route_idx]
        del solution.request_to_route[req]
        route.remove_request(req)
        solution.unserved_requests.add(req)

    return removed

def heavy_request_removal(solution, q):
    # Collect all served requests
    served = list(solution.request_to_route.keys())

    # Sort by descending demand
    served.sort(
        key=lambda r: solution.instance.requests[r - 1]["demand"],# * a(solution.instance.requests_location_array[r-1]["drop_off"], solution.instance.requests_location_array[r-1]["pick_up"]),
        reverse=True
    )

    removed = set()

    for req in served[:q]:
        route_idx = solution.request_to_route[req]
        route = solution.routes[route_idx]

        route.remove_request(req)
        del solution.request_to_route[req]
        solution.unserved_requests.add(req)

        removed.add(req)

    return removed


# === [ Repair OPERATORS ] ===
def greedy_repair(solution, max_gap=5):
    while len(solution.unserved_requests) > solution.instance.n - solution.instance.gamma:
        best = None
        for req in list(solution.unserved_requests):
            for route_idx, route in enumerate(solution.routes):
                # we have to change the logic of this insertion operator: instead of iterating over all p in range(L + 1), we only try to insert if not route.load_profile[p] + req["demand"] > instance.C
                best_delta = None
                best_pos = None

                L = len(route.nodes)
                load_profile = route.load_profile
                load_profile.append(0)

                for p in range(L + 1):
                    if load_profile[p] + solution.instance.requests[req - 1]["demand"] > solution.instance.C:
                        #print(f"skipping position {p} in route {route_idx} for request {req} becaues its demand {instance.requests[req + 1]['demand']} is too high")
                        continue
                    for d in range(p + 1, min(p + max_gap, L + 2)):
                        delta = route.try_insert_request(req, p, d)
                        if delta is not None:
                            if best_delta is None or delta < best_delta:
                                best_delta = delta
                                best_pos = (p, d)
                            if best_pos is not None:
                                if best is None or delta < best[0]:
                                    best = (delta, req, route_idx, best_pos)

                    if best is None:
                        break

        _, req, route_idx, (p, d) = best
        solution.routes[route_idx].insert_request(req, p, d)
        solution.unserved_requests.remove(req)
        solution.request_to_route[req] = route_idx


def regret_3_repair(solution, max_gap = 5):
    while len(solution.unserved_requests) > solution.instance.n - solution.instance.gamma: # while we don't serve at least gamma requests
        best_req = None
        best_regret = -float("inf")
        best_route = None

        for req in solution.unserved_requests:
            costs = []  # collect the "cost" of each unserved request
            #print(f"req: {req}")

            for route in solution.routes:
                # try_insert_request(self, req_id, pick_up_idx_pos, drop_off_idx_pos):
                L = len(route.nodes)
                load_profile = route.load_profile
                load_profile.append(0)
                for p in range(L + 1):
                    if load_profile[p] + solution.instance.requests[req - 1]["demand"] > solution.instance.C:
                        #print(f"skipping position {p} in route {route_idx} for request {req} becaues its demand {instance.requests[req + 1]['demand']} is too high")
                        continue
                    for d in range(p + 1, min(p + max_gap, L + 2)):
                        delta = route.try_insert_request(req,p, d)
                        if delta is not None:
                            costs.append((delta, route, p, d)) # the "cost" is the delta in the objectieve function we would achieve by inserting at this position

            if len(costs) < 1: # we skip the sorting it there are not at least two entries
                continue

            costs.sort(key=lambda x: x[0]) # we sort the costs per delta
            c1 = costs[0][0] # cost of the best solution
            c3 = costs[2][0] if len(costs) >= 3 else costs[-1][0] # choose the second on eor else the third one if we don't have enough entries
            regret = c3 - c1 # regret is always positive, hsa the largest possible value if c1 is really _much_ better than c3, -> tells us "how much" c1 is better than c3

            if regret > best_regret: # instead of inserting the request where we would get instantly most optimal objective score, we insert it where the difference in the objective score to the other possiblilities is the highest
                best_regret = regret
                best_req = req
                best_route = costs[0][1]
                best_p = costs[0][2]
                best_d = costs[0][3]

        if best_req is None:
            raise Exception("No best insertion found")
            break

        # insert_request(self, req_id, p_pos, d_pos):
        best_route.insert_request(best_req, best_p, best_d)
        solution.unserved_requests.remove(best_req)
        route_idx = solution.routes.index(best_route)
        solution.request_to_route[best_req] = route_idx


class OperatorPool:
    def __init__(self, operators, min_weight=1e-6):
        self.ops = operators
        self.weights = [1.0] * len(operators)
        self.applied = [0] * len(operators)
        self.success = [0] * len(operators)
        self.min_weight = min_weight

    def select(self):
        return random.choices(
            range(len(self.ops)),
            weights=self.weights,
            k=1
        )[0]

    def record_application(self, idx):
        self.applied[idx] += 1

    def record_success(self, idx):
        self.success[idx] += 1

    def adapt(self, gamma=0.1):
        for i in range(len(self.weights)):
            if self.applied[i] > 0:
                performance = self.success[i] / self.applied[i]
            else:
                performance = 0.0

            self.weights[i] = (
                (1 - gamma) * self.weights[i] + gamma * performance
            )

            # enforce positivity
            self.weights[i] = max(self.weights[i], self.min_weight)

            # reset counters
            self.applied[i] = 0
            self.success[i] = 0

@dataclass
class ALNSParams:
    T0: float
    cooling: float
    destroy_fraction: int
    rho: float
    reward_best: float
    reward_accept: float
    reward_reject: float
    regret_k: int
    max_gap: int



def alns(instance, initial_solution, params, iters=100, log_file="alns_log.csv"):

    def accept(new, current, temperature):
        if new.total_cost < current.total_cost:
            return True
        delta = new.total_cost - current.total_cost
        return random.random() < math.exp(-delta / temperature)

    destroy_ops = OperatorPool([
        lambda s: random_removal(s, q=params.destroy_fraction),
        lambda s: worst_removal(s, q=params.destroy_fraction),
        lambda s: shaw_removal(s, q=params.destroy_fraction),
        lambda s: heavy_request_removal(s, q= params.destroy_fraction)
    ])

    repair_ops = OperatorPool([
        lambda s: greedy_repair(s, max_gap=params.max_gap),
        lambda s: regret_3_repair(s, max_gap=params.max_gap),
    ])

    current = initial_solution
    best = current.copy()
    T = params.T0

    # Open CSV file
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        # CSV header
        writer.writerow([
            "iteration",
            "objective",
            "destroy_op",
            "repair_op",
            "accepted",
            "new_best",
            "time_sec",
            "temperature"
        ])
        for it in range(iters):
            t0 = time.perf_counter()
            new = current.copy()

            d = destroy_ops.select()
            destroy_ops.record_application(d)
            r = repair_ops.select()
            repair_ops.record_application(r)

            destroy_ops.ops[d](new)
            repair_ops.ops[r](new)
            new.recompute_cost()

            if not new.is_feasible():
                continue

            accepted = False
            new_best = False

            if accept(new, current, T):
                accepted = True
                current = new
                reward = params.reward_accept
                destroy_ops.record_success(d)
                repair_ops.record_success(r)

                if new.total_cost < best.total_cost:
                    best = new.copy()
                    reward = params.reward_best
                    new_best = True
            else:
                reward = params.reward_reject

            #destroy_ops.update(d, reward)
            #repair_ops.update(r, reward)

            elapsed = time.perf_counter() - t0

            # Write CSV row
            writer.writerow([
                it,
                new.total_cost,
                d,
                r,
                accepted,
                new_best,
                elapsed,
                T
            ])

            if it % 10 == 0:
                destroy_ops.adapt()
                repair_ops.adapt()
                T *= params.cooling
    return best
