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

# used for delta evaluation later on
def delta_objective(instance, old_routes, new_routes):
    changed = []
    for i, (r_old, r_new) in enumerate(zip(old_routes, new_routes)):
        if r_old != r_new:
            changed.append(i)

    d_old = []
    d_new = []
    for i in changed:
        d_old.append(get_total_distance(instance, old_routes[i]))
        d_new.append(get_total_distance(instance, new_routes[i]))

    delta_dist = sum(d_new) - sum(d_old)

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

    return delta_dist + delta_fairness

def plot_routes(instance, routes):
    plt.figure(figsize=(10, 10))
    plt.scatter(*instance.depot, c="black", marker="s", s=100, label="Depot")  # draw the depot

    # Draw each vehicle's route
    for i, route in enumerate(routes):
        color = f"C{i % 10}"  # use matplotlib color cycle
        x_coords = [instance.depot[0]]
        y_coords = [instance.depot[1]]

        for idx in route:
            loc = instance.requests_location_array[idx - 1]
            x_coords.append(loc[0])
            y_coords.append(loc[1])

        x_coords.append(instance.depot[0])
        y_coords.append(instance.depot[1])

        plt.plot(x_coords, y_coords, "-", color=color, label=f"Vehicle {i+1}")

        # Mark pickups (▲) and drop-offs (▼)
        for idx in route:
            loc = instance.requests_location_array[idx - 1]
            if idx <= instance.n:  # pickup
                plt.scatter(loc[0], loc[1], marker="^", color=color, s=70)
            else:  # drop-off
                plt.scatter(loc[0], loc[1], marker="v", color=color, s=70)

    # Compute metrics
    total_distance = sum(get_total_distance(instance, route) for route in routes)
    jain_score = get_Jain_fairness(instance, routes)
    obj_value = objective_function(instance, routes)

    # Add metrics to the plot
    plt.figtext(0.5, -0.01,  # x=0.5 centers it, y=-0.02 slightly below the plot
                f"Total Distance: {total_distance:.2f}    "
                f"Jain Fairness: {jain_score:.3f}    "
                f"Objective: {obj_value:.2f}",
                ha="center", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.show()

def nearest_neighbor_heuristic(instance):
    open_requests = instance.requests.copy()
    routes = [[] for _ in range(instance.n_k)]
    current_vehicle_location = [instance.depot for _ in range(instance.n_k)]
    current_vehicle_capacity = [0 for _ in range(instance.n_k)]
    current_vehicle_requests = [[] for _ in range(instance.n_k)]

    n_served = 0
    while n_served < instance.gamma:
        for k in range(instance.n_k):
            if not open_requests and not current_vehicle_requests[k]:
                continue  # we skip this vehicle
            # print(f"lenopenreq: {len(open_requests)}")
            nearest_pick_up_location = None
            if open_requests:
                try:
                    nearest_pick_up_location = min(open_requests,
                                                   key=lambda req: a(req["pick_up"], current_vehicle_location[k]))
                except ValueError:  # we just don't have any open requests anymore
                    # print("no more open requests")
                    pass
            if nearest_pick_up_location is not None and (
                    nearest_pick_up_location["demand"] + current_vehicle_capacity[k] <= instance.C):
                # print(f"PICKING UP: {nearest_pick_up_location}")
                # print(nearest_pick_up_location["demand"] + current_vehicle_capacity[k])
                current_vehicle_location[k] = nearest_pick_up_location[
                    "pick_up"]  # we are driving to the new pick up location
                routes[k].append(nearest_pick_up_location["index"])  # write it down in our route list
                current_vehicle_requests[k].append(nearest_pick_up_location)  # take the request
                current_vehicle_capacity[k] += nearest_pick_up_location["demand"]  # use some loading capacity
                open_requests.remove(nearest_pick_up_location)  # it's not an open request anymore
            elif current_vehicle_requests[k]:
                nearest_drop_off_location = min(current_vehicle_requests[k],
                                                key=lambda req: a(req["drop_off"], current_vehicle_location[k]))
                # print(f"DROPPING OFF: {nearest_drop_off_location}")
                current_vehicle_location[k] = nearest_drop_off_location[
                    "drop_off"]  # driving to the drop off location
                routes[k].append(
                    nearest_drop_off_location["index"] + instance.n)  # writing down the location index in our route
                current_vehicle_requests[k].remove(nearest_drop_off_location)  # not our request anymore
                current_vehicle_capacity[k] -= nearest_drop_off_location["demand"]  # freeing some loading capacity
                n_served += 1  # we have succesfully completed one request!
            else:
                print("some other case occured")
    # drop off the loaded requests:
    while any(current_vehicle_requests):
        for k in range(instance.n_k):
            if current_vehicle_requests[k]:
                nearest_drop_off_location = min(current_vehicle_requests[k],
                                                key=lambda req: a(req["drop_off"], current_vehicle_location[k]))
                current_vehicle_location[k] = nearest_drop_off_location[
                    "drop_off"]  # driving to the drop off location
                routes[k].append(
                    nearest_drop_off_location["index"] + instance.n)  # writing down the location index in our route
                current_vehicle_requests[k].remove(nearest_drop_off_location)  # not our request anymore
                current_vehicle_capacity[k] -= nearest_drop_off_location["demand"]  # freeing some loading capacity
                n_served += 1  # we have succesfully completed one request!
    return routes