import os
from src.misc import *
from src.route import Route
from src.instance import Instance
import copy

class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.routes = []

        self.unserved_requests = set(r["index"] for r in instance.requests)

        self.total_cost = 0.0
        self.total_dist = 0.0
        self.fairness = 0.0

        # Fast lookup
        self.request_to_route = {} # tells us in which route we can find which request

    def load_from_arrays(self, routes):
        self.total_cost, self.total_dist, self.fairness = 0.0, 0.0, 0.0 # resetting everything
        self.unserved_requests = set(r["index"] for r in self.instance.requests)
        self.routes = []
        assert len(routes) == self.instance.n_k
        for r_idx, r in enumerate(routes):
            route = Route(self.instance)
            route.load_from_array(r)
            self.routes.append(route)
            self.total_dist += route.distance
            for req in route.requests:
                self.unserved_requests.remove(req)
                self.request_to_route[req] = r_idx
        self.fairness = get_Jain_fairness(self.instance, routes)
        self.total_cost = objective_function(self.instance, routes)

    def copy(self):
        return copy.deepcopy(self)

    def recompute_cost(self):
        self.total_cost, self.total_dist, self.fairness = 0.0, 0.0, 0.0 # resetting everything
        self.unserved_requests = set(r["index"] for r in self.instance.requests)
        #self.request_to_route = {} ToDo: reload requests_to_route!
        routes_as_array = []
        for route in self.routes:
            route.recompute()
            self.total_dist += route.distance
            routes_as_array.append(route.nodes)
            for req in route.requests:
                self.unserved_requests.remove(req)
        self.fairness = get_Jain_fairness(self.instance,routes_as_array)
        self.total_cost = objective_function(self.instance, routes_as_array)
        assert self.is_feasible()
        assert len(self.routes) == self.instance.n_k

    def is_feasible(self):
        # checking first condition - The vehicle capacity must never be exceeded at any point along the route
        # since all routes are already checked, we can assume this is true
        # checking second condition -  Each served request must be handled in its entirety by a single vehicle.
        # based on the design of the Route class, we also assume this is true
        # thirdly, at least gamma requests have to be served
        if len(self.unserved_requests) + self.instance.gamma <= self.instance.n:
            return True
        else:
            return False

    def plot(self):
        routes_as_array = [r.nodes for r in self.routes]
        plot_routes(self.instance, routes_as_array)

    def save_to_file(self, instance_file_path, output_dir="solutions"):
        """
        Save solution in the assignment-specified format.

        Format:
        <instance_name>
        R1,1 R1,2 ... R1,|R1|
        R2,1 R2,2 ... R2,|R2|
        ...
        """

        os.makedirs(output_dir, exist_ok=True)

        # extract instance name (no path, no extension)
        instance_name = os.path.splitext(os.path.basename(instance_file_path))[0]

        out_file = os.path.join(output_dir, instance_name + "_solution.txt")

        with open(out_file, "w") as f:
            # first line: instance name
            f.write(instance_name + "\n")

            # each route on one line
            for route in self.routes:
                # IMPORTANT: adapt this depending on your Route structure
                # This should be the ordered list of request-location indices
                nodes = route.nodes

                if len(nodes) == 0:
                    f.write("\n")
                    continue

                line = " ".join(str(node) for node in nodes)
                f.write(line + "\n")

        print(f"Saved solution to {out_file}")