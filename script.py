# does everything as in main.ipynb, but as a python script instead og a jupyter notebook, so that we can run it on the cluster more easily

# === IMPORTS ===
from dataclasses import dataclass
import math
from random import random
from collections import deque
import csv
import time
import pandas as pd
import os, glob
from tqdm import tqdm



import matplotlib.pyplot as plt
import random
random.seed(1090)

# === DEFINING INSTANCE CLASS ===
@dataclass
class Instance:
    def __init__(self, filename: str | None = None):
        if filename is not None:
            self.load_from_file(filename)

    @staticmethod # tell python this method doesn't depend on "self"
    def read_tuple(f):
        return tuple(map(int, f.readline().split()))

    def load_from_file(self, filename):
        with open(filename) as input_file:
            params_line = input_file.readline().split(" ")
            self.n = int(params_line[0])
            self.n_k = int(params_line[1])
            self.C = int(params_line[2])
            self.gamma = int(params_line[3])
            self.rho = float(params_line[4])
            _ = input_file.readline()
            demand_line = [int(c_i) for c_i in input_file.readline().split()]
            _ = input_file.readline()
            self.depot = self.read_tuple(input_file)
            self.requests = []
            self.requests_location_array = []
            for i in range(self.n):
                pick_up = self.read_tuple(input_file)
                request = {
                    "pick_up": pick_up,
                    "drop_off": None,
                    "demand": demand_line[i],
                    "index": i + 1 # because they start with one (not zero) in the assignment
                }
                self.requests.append(request)
                self.requests_location_array.append(pick_up)
            for i in range(self.n):
                drop_off = self.read_tuple(input_file)
                self.requests[i]["drop_off"] = drop_off
                self.requests_location_array.append(drop_off)
            if input_file.readline().strip() != "":# make sure we are at the end of the file and there's no more content
                raise ValueError("Unexpected extra content at end of file")

# === DEFINING HELPER CLASS ===
# takes two tuples representing carthesian coordinates as input
def a(u, v):
    return math.ceil(math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2))

# takes the route as list of indices of request locations as input
def get_total_distance(instance, route):
    total_distance = 0
    current_location = instance.depot # we start at the depot
    for location_index in route:
        location_index = location_index - 1 # be careful with off-by-one errors!
        next_location = instance.requests_location_array[location_index]
        total_distance = total_distance + a(current_location, next_location)
        current_location = next_location
    total_distance = total_distance + a(current_location, instance.depot)
    return total_distance

# take a list of routes as input
def get_Jain_fairness(instance, routes):
    numerator = 0
    denominator = 0
    for route in routes:
        d = get_total_distance(instance, route)
        numerator = numerator + d
        denominator = denominator + d**2
    return numerator**2 / (instance.n_k * denominator) # I know it says n in the assignment but n_k makes more sense and its like that on wikipedia

def objective_function(instance, routes):
    value = 0
    for route in routes:
        value = value + get_total_distance(instance, route)
    value = value + instance.rho * (1 - get_Jain_fairness(instance, routes))
    return value

def delta_objective(instance, old_routes, new_routes):
    """
    Compute Δf = f(new_routes) - f(old_routes) in O(k) time,
    where k is the number of routes that changed (normally 1 or 2).
    """
    # --- 1. Identify changed routes ---
    changed = []
    for i, (r_old, r_new) in enumerate(zip(old_routes, new_routes)):
        if r_old != r_new:
            changed.append(i)

    # --- 2. Compute old and new distances only for changed routes ---
    d_old = []
    d_new = []
    for i in changed:
        d_old.append(get_total_distance(instance, old_routes[i]))
        d_new.append(get_total_distance(instance, new_routes[i]))

    # --- 3. Compute Δ total distance ---
    delta_dist = sum(d_new) - sum(d_old)

    # --- 4. Compute Jain fairness terms ---
    # Old aggregated sums
    old_all_dist = []
    for r in old_routes:
        old_all_dist.append(get_total_distance(instance, r))
    old_sum = sum(old_all_dist)
    old_sum_sq = sum(d * d for d in old_all_dist)

    # New aggregated sums (update only the changed routes)
    new_all_dist = old_all_dist.copy()
    for i, idx in enumerate(changed):
        new_all_dist[idx] = d_new[i]    # replace old distance with new

    new_sum = sum(new_all_dist)
    new_sum_sq = sum(d * d for d in new_all_dist)

    fairness_old = (old_sum * old_sum) / (instance.n_k * old_sum_sq)
    fairness_new = (new_sum * new_sum) / (instance.n_k * new_sum_sq)

    delta_fairness = instance.rho * ((1 - fairness_new) - (1 - fairness_old))

    # --- 5. Full delta ---
    return delta_dist + delta_fairness


# === PILOT SEARCH ===
# creating an own class to represent the nodes
@dataclass
class Node:
    routes: list           # list[list[int]]
    loc: list              # current location per vehicle
    capacity: list              # current used capacity
    onboard: list          # onboard[k] = list of requests
    open_reqs: list        # list of remaining requests
    n_served: int
    score: float           # evaluation score

    def copy(self):
        """Create a deep copy of the state, but without copying the underlying request dicts."""
        return Node(
            routes=[r.copy() for r in self.routes],
            loc=self.loc.copy(),
            capacity=self.capacity.copy(),
            onboard=[lst.copy() for lst in self.onboard],
            open_reqs=self.open_reqs.copy(),
            n_served=self.n_served,
            score=self.score
        )
# create the children nodes
def expand_node(instance, node):
    if node.open_reqs == [] and all(len(o)==0 for o in node.onboard):
        print("No possible moves in this state, served: ", node.n_served)
    successors = []
    for k in range(instance.n_k):
        # Pick up successors
        for req in node.open_reqs:
            if node.capacity[k] + req["demand"] <= instance.C:
                S2 = node.copy()
                old_loc = S2. loc[k] # move old vehicle
                new_loc = req["pick_up"]
                S2.loc[k] = new_loc
                S2.routes[k].append(req["index"])
                S2.onboard[k].append(req)
                S2.open_reqs.remove(req)
                S2.capacity[k] += req["demand"]
                S2.score  = objective_function(instance, S2.routes)
                successors.append(S2)
        # Drop off successors
        for req in node.onboard[k]:
            S2 = node.copy()
            old_loc = S2.loc[k]
            new_loc = req["drop_off"]
            S2.loc[k] = new_loc
            S2.routes[k].append(req["index"] + instance.n)
            S2.onboard[k].remove(req)
            S2.capacity[k] -= req["demand"]
            S2.score = objective_function(instance, S2.routes)
            S2.n_served += 1
            successors.append(S2)
    return successors

def flush_dropoffs(instance, node):
    for k in range(instance.n_k):
        while node.onboard[k]: # as long as we have still requests in the vehicle
            req = node.onboard[k][0] # take the first request on board of the vehicle
            node.loc[k] = req["drop_off"] # set the vehicle location to the drop of point
            node.routes[k].append(req["index"] + instance.n) # appending drop off location to route
            node.capacity[k] -= req["demand"] # decreasing capacity
            node.onboard[k].remove(req)
            node.n_served += 1
    return node

def beam_search(instance, beta):
    init = Node(
        routes = [[] for _ in range(instance.n_k)],
        loc = [instance.depot for _ in range(instance.n_k)],
        capacity = [0 for _ in range(instance.n_k)],
        onboard=[[] for _ in range(instance.n_k)],
        open_reqs=instance.requests.copy(),
        n_served=0,
        score=0.0
    )
    beam = [init]
    while True:
        if all(s.n_served >= instance.gamma for s in beam):
            out = min(successors, key=lambda s: s.score)
            for k in range(instance.n_k): # delivering the remaining drop offs
                while out.onboard[k]: # as long as we have still requests in the vehicle
                    req = out.onboard[k][0] # take the first request on board of the vehicle
                    out.loc[k] = req["drop_off"] # set the vehicle location to the drop of point
                    out.routes[k].append(req["index"] + instance.n) # appending drop off location to route
                    out.capacity[k] -= req["demand"] # decreasing capacity
                    out.onboard[k].remove(req)
                    out.n_served += 1
            return out.routes
        successors = []
        for s in beam:
            successors.extend(expand_node(instance, s))

        successors.sort(key=lambda s: s.score)
        beam = successors[:beta]


# === actually generating solutions ===
def save_solution(instance_file_path, routes, output_dir="solutions"):
    """
    Saves the solution in the required format:
    first line = instance file name (without path & extension)
    next lines = each route (only request indices), space-separated
    """

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # extract instance name
    instance_name = os.path.splitext(os.path.basename(instance_file_path))[0]

    # prepare output file path
    out_file = os.path.join(output_dir, instance_name + "_solution.txt")

    with open(out_file, "w") as f:
        # first line: instance name
        f.write(instance_name + "\n")

        # then each vehicle route, request indices separated with spaces
        for route in routes:
            # route is something like [1, 11, 2, 3, 13, ...]
            line = " ".join(str(r) for r in route)
            f.write(line + "\n")

    print(f"Saved solution to {out_file}")

# Lets generate a solution
relevant_files = ["instances/100/competition/instance61_nreq100_nveh2_gamma91.txt",  "instances/1000/competition/instance61_nreq1000_nveh20_gamma879.txt","instances/2000/competition/instance61_nreq2000_nveh40_gamma1829.txt"]
for file in [relevant_files[0]]:
    instance = Instance(file)
    solution = beam_search(instance, 50)
    #solution = beam_search(instance, 3)
    save_solution(file, solution, "solutions_deterministic_construction_heuristic")