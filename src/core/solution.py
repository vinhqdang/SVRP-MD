from dataclasses import dataclass, field
from typing import List
from .route import Route

@dataclass
class Solution:
    routes: List[Route]
    total_distance: float = 0.0
    expected_penalty: float = 0.0
    n_vehicles: int = 0
    
    # Set after evaluation
    objective: float = float('inf')
    
    def compute_objective(self, c_f: float, c_p: float):
        self.n_vehicles = len(self.routes)
        self.objective = self.total_distance + c_f * self.n_vehicles \
                         + c_p * self.expected_penalty
        return self.objective

    def evaluate(self, instance):
        from src.oracle.route_eval import eval_route
        self.total_distance = 0.0
        self.expected_penalty = 0.0
        for r in self.routes:
            curr = 0
            for c in r.customers:
                self.total_distance += instance.distance[curr][c]
                curr = c
            self.total_distance += instance.distance[curr][0]
            self.expected_penalty += eval_route(r, instance)
        self.n_vehicles = len(self.routes)
        self.objective = self.total_distance + instance.cost_fleet * self.n_vehicles \
                         + instance.cost_penalty * self.expected_penalty
        return self.objective
