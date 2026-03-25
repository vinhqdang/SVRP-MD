from dataclasses import dataclass
from typing import List

@dataclass
class Route:
    customers: List[int]    # ordered list of customer indices (1-based)
    # Note: orientation is determined at evaluation time (best of two)
    
    def reverse(self) -> 'Route':
        return Route(self.customers[::-1])
    
    def __len__(self): return len(self.customers)
    def __iter__(self): return iter(self.customers)
