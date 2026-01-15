from src.misc import *
from src.instance import Instance

class Route:
    def __init__(self, instance):
        self.instance = instance
        self.capacity = instance.C

        self.nodes = []              # sequence of pickup/delivery node IDs
        self.requests = set()        # request IDs served by this route

        self.load_profile = []      # load after each node
        self.min_slack = instance.C    # min remaining capacity (thightest point along the route)
        self.distance = 0.0          # route length

    def recompute(self):
        load = 0
        self.load_profile = []
        self.min_slack = self.capacity

        for node in self.nodes:
            if node <= self.instance.n:
                load += self.instance.requests[node - 1]["demand"]
            else:
                load -= self.instance.requests[node - self.instance.n - 1]["demand"]

            if load < 0 or load > self.capacity:
                #raise ValueError("Infeasible route")
                pass

            self.load_profile.append(load)
            self.min_slack = min(self.min_slack, self.capacity - load)
        self.distance = get_total_distance(self.instance, self.nodes)

    def load_from_array(self, route):
        # this function assumes valid input, because the input will mainly come from algorithms that are already tested and tried out
        self.nodes = route
        self.requests = {i for i in route if 1 <= i <= self.instance.n}
        self.recompute()

    def copy(self):
        return copy.deepcopy(self)

    def remove_request(self, req_id):
        if not req_id in self.requests:
            raise Exception("req_id not in this route!")
        self.nodes = [n for n in self.nodes if n != req_id and n != (req_id + self.instance.n)]
        self.requests.remove(req_id)
        self.recompute()
        #print("removed request ", req_id)

    def try_insert_request(self, req_id, pick_up_idx_pos, drop_off_idx_pos):
        # we start counting with 0!!!!
        if req_id > self.instance.n or req_id in self.requests:
            raise Exception("Invalid req id!")
            return None
        if drop_off_idx_pos < pick_up_idx_pos:
            raise Exception("drop off before pick up!")
            return None
        demand = self.instance.requests[req_id - 1]["demand"]
        C = self.capacity
        load = self.load_profile[pick_up_idx_pos - 1] if pick_up_idx_pos > 0 else 0
        if load + demand > C:
            return None
        for i in range(pick_up_idx_pos, drop_off_idx_pos):
            if self.load_profile[i] + demand > C:
                return None
        # use delta evaluation for cost estimation
        if pick_up_idx_pos > len(self.nodes):
            #print("Warning: pick up  index too high")
            pick_up_idx_pos = len(self.nodes)
            return None
        A = self.instance.requests_location_array[self.nodes[pick_up_idx_pos - 1] - 1] if pick_up_idx_pos > 0 else self.instance.depot # stop before inserting
        B = self.instance.requests_location_array[self.nodes[pick_up_idx_pos] - 1] if pick_up_idx_pos < len(self.nodes) else self.instance.depot # stop after inserting

        if pick_up_idx_pos != drop_off_idx_pos:
            delta = (
                a(A, self.instance.requests_location_array[req_id - 1])
                + a(self.instance.requests_location_array[req_id - 1], B)
                - a(A, B)
            )
        else:
            delta = (
                a(A, self.instance.requests_location_array[req_id - 1])
                - a(A, B)
            )

        d_adj = drop_off_idx_pos

        if d_adj > len(self.nodes):
            d_adj = len(self.nodes)
            #print("Warning: drop of position too high")
            return None
        Cn = self.instance.requests_location_array[self.nodes[d_adj - 1] - 1] if d_adj > 0 else self.instance.depot
        En = self.instance.requests_location_array[self.nodes[d_adj] - 1] if d_adj < len(self.nodes) else self.instance.depot

        if pick_up_idx_pos != drop_off_idx_pos:
            delta += (
                a(Cn, self.instance.requests_location_array[req_id + self.instance.n - 1])
                + a(self.instance.requests_location_array[req_id + self.instance.n - 1], En)
                - a(Cn, En)
            )
        else:
            delta += (
                a(self.instance.requests_location_array[req_id - 1], self.instance.requests_location_array[req_id - 1 + self.instance.n])
                + a(self.instance.requests_location_array[req_id - 1 + self.instance.n], En)
            )
        return delta

    def insert_request(self, req_id, p_pos, d_pos):
        #print(f"Picking up req {req_id} at {p_pos} and dropping it off at {d_pos}")
        delta = self.try_insert_request(req_id, p_pos, d_pos)
        if delta is not None:
            pickup = req_id
            dropoff = req_id + self.instance.n

            self.nodes.insert(p_pos, pickup)
            self.nodes.insert(d_pos + 1, dropoff)
            self.requests.add(req_id)
            self.recompute()

    def marginal_cost(self, req_id):
        original_distance = self.distance
        temp_nodes = [n for n in self.nodes if n != req_id and n != req_id  + self.instance.n]
        new_distance = get_total_distance(self.instance, temp_nodes)
        return original_distance - new_distance
