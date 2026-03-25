from dataclasses import dataclass, field
import numpy as np
import json

@dataclass
class Instance:
    name: str
    n_customers: int            # |V+|
    capacity: float             # Q
    initial_load: float         # L_0 (default 0.0)
    cost_penalty: float         # c_p  (penalty multiplier)
    cost_fleet: float           # c_f  (per-vehicle cost)
    
    # Graph
    distance: np.ndarray        # shape (n+1, n+1), index 0 = depot
    
    # Demands — shape (N_scenarios, n_customers)
    # demand[xi, i] = realisation of d_i under scenario xi
    demand: np.ndarray
    prob: np.ndarray            # shape (N_scenarios,), sum to 1.0
    mean_demand: np.ndarray = field(init=False)     # shape (n_customers,)  pre-computed
    var_demand: np.ndarray = field(init=False)      # shape (n_customers,)  pre-computed

    def __post_init__(self):
        assert self.demand.shape[0] == len(self.prob)
        assert abs(self.prob.sum() - 1.0) < 1e-9
        self.mean_demand = (self.prob[:, None] * self.demand).sum(axis=0)
        self.var_demand  = (self.prob[:, None] * self.demand**2).sum(axis=0) \
                           - self.mean_demand**2

    @property
    def n_scenarios(self): return len(self.prob)
    
    @property
    def n_nodes(self): return self.n_customers + 1  # includes depot

def load_instance(path: str) -> Instance:
    """Load from JSON format (see Section 6 for schema)."""
    with open(path) as f:
        d = json.load(f)
    return Instance(
        name=d['name'],
        n_customers=d['n_customers'],
        capacity=d['capacity'],
        initial_load=d.get('initial_load', 0.0),
        cost_penalty=d['cost_penalty'],
        cost_fleet=d['cost_fleet'],
        distance=np.array(d['distance']),
        demand=np.array(d['demand']),     # shape (N_scenarios, n_customers)
        prob=np.array(d['prob']),
    )
