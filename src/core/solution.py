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
