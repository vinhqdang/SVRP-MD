import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from src.core.solution import Solution
from src.core.route import Route

def is_cycle_robust_feasible(customers, instance) -> bool:
    scenarios = instance.demand
    
    forward_feasible = True
    for s in range(instance.n_scenarios):
        load = instance.initial_load
        peak = load
        for c in customers:
            load += scenarios[s, c-1]
            peak = max(peak, load)
        if peak > instance.capacity:
            forward_feasible = False
            break
            
    backward_feasible = True
    for s in range(instance.n_scenarios):
        load = instance.initial_load
        peak = load
        for c in reversed(customers):
            load += scenarios[s, c-1]
            peak = max(peak, load)
        if peak > instance.capacity:
            backward_feasible = False
            break
            
    return forward_feasible or backward_feasible

class BaselineCallback:
    def __init__(self, instance, x_vars):
        self.instance = instance
        self.x_vars = x_vars
        self.cut_counts = {'route_capacity': 0, 'sec': 0}

    def callback(self, model, where):
        self.model = model
        self.where = where
        if where == GRB.Callback.MIPSOL:
            x_val = {k: model.cbGetSolution(v) for k, v in self.x_vars.items()}
            routes, subtours = self._extract_routes_and_subtours(x_val)
            
            for comp in subtours:
                self._add_sec(list(comp))
                
            if subtours:
                return
                
            for r in routes:
                if not is_cycle_robust_feasible(r.customers, self.instance):
                    self._add_route_capacity_cut(r)
                    
    def _extract_routes_and_subtours(self, x_val):
        edges = [(i, j) for (i, j), val in x_val.items() if val > 0.5]
        G = nx.MultiGraph()
        for (i, j), val in x_val.items():
            for _ in range(int(round(val))):
                G.add_edge(i, j)
                
        routes = []
        if 0 in G:
            for nbr in list(G.neighbors(0)):
                if G.has_edge(0, nbr):
                    G.remove_edge(0, nbr)
                    curr = nbr
                    path = []
                    while curr != 0:
                        path.append(curr)
                        next_nodes = list(G.neighbors(curr))
                        if not next_nodes:
                            break
                        nxt = next_nodes[0]
                        G.remove_edge(curr, nxt)
                        curr = nxt
                    routes.append(Route(path))
                    
        subtours = [comp for comp in nx.connected_components(G) if 0 not in comp and len(comp) > 1]
        return routes, subtours

    def _add_sec(self, S):
        x_S = gp.quicksum(self.x_vars[min(i,j), max(i,j)] for i in S for j in S if i < j)
        self.model.cbLazy(x_S <= len(S) - 1)
        self.cut_counts['sec'] += 1

    def _add_route_capacity_cut(self, route):
        customers = route.customers
        edges = []
        curr = 0
        for c in customers:
            edges.append((min(curr, c), max(curr, c)))
            curr = c
        edges.append((min(curr, 0), max(curr, 0)))
        
        rhs_activation = gp.quicksum(self.x_vars[e] for e in edges)
        
        self.model.cbLazy(rhs_activation <= len(edges) - 1)
        self.cut_counts['route_capacity'] += 1

def build_baseline_master(instance, n_vehicles_max: int = 10):
    m = gp.Model("CVRP_Robust")
    n = instance.n_customers
    
    x = {}
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            upper = 2 if i == 0 else 1
            x[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=upper, name=f"x_{i}_{j}")
    
    for i in range(1, n + 1):
        m.addConstr(
            gp.quicksum(x[min(i,j), max(i,j)] for j in range(n + 1) if j != i) == 2,
            name=f"degree_{i}"
        )
    
    m.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n + 1)) <= 2 * n_vehicles_max,
        name="depot_degree"
    )
    
    dist_cost = gp.quicksum(
        instance.distance[i][j] * x[min(i,j), max(i,j)]
        for i in range(n + 1) for j in range(i + 1, n + 1)
    )
    fleet_cost = instance.cost_fleet * gp.quicksum(x[0, j] for j in range(1, n + 1)) / 2.0
    
    m.setObjective(dist_cost + fleet_cost, GRB.MINIMIZE)
    m.update()
    
    return m, x

def solve_baseline_robust(
    instance,
    time_limit_s: float = 3600.0,
    verbose: bool = False
) -> dict:
    model, x_vars = build_baseline_master(instance)
    
    cb = BaselineCallback(instance, x_vars)
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_limit_s
    model.Params.Threads = 1
    if not verbose:
        model.Params.OutputFlag = 0
    
    model.optimize(cb.callback)
    
    result = {
        'objective': model.ObjVal if model.SolCount > 0 else float('inf'),
        'gap': model.MIPGap * 100 if model.SolCount > 0 else 100.0,
        'n_nodes': int(model.NodeCount),
        'solve_time_s': model.Runtime,
        'cuts_added': cb.cut_counts,
        'proved_optimal': model.Status == GRB.OPTIMAL
    }
    
    from src.bnc.master import _extract_solution
    if model.SolCount > 0:
        result['solution'] = _extract_solution(model, x_vars, instance)
    else:
        result['solution'] = None
    return result
