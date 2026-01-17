import numpy as np
import random
import copy
import csv
import time
from src.solution import Solution
from src.misc import a, capacity_feasible, objective_function
from dataclasses import dataclass


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