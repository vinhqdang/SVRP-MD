import gurobipy as gp
from gurobipy import GRB
from src.core.solution import Solution
from src.core.route import Route
from src.bnc.callback import SolverCallback

def build_master(instance, n_vehicles_max: int = 10):
    m = gp.Model("SVRP_MD")
    n = instance.n_customers
    
    x = {}
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            upper = 2 if i == 0 else 1
            x[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=upper,
                                name=f"x_{i}_{j}")
    
    theta = {i: m.addVar(lb=0.0, name=f"theta_{i}")
             for i in range(1, n + 1)}
    
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
    recourse_cost = instance.cost_penalty * gp.quicksum(theta.values())
    
    fleet_cost = instance.cost_fleet * gp.quicksum(x[0, j] for j in range(1, n + 1)) / 2.0
    
    m.setObjective(dist_cost + recourse_cost + fleet_cost, GRB.MINIMIZE)
    m.update()
    
    return m, x, theta

def _extract_solution(model, x_vars, instance):
    import networkx as nx
    n = instance.n_customers
    edges = []
    for (i, j), var in x_vars.items():
        val = int(round(var.x))
        for _ in range(val):
            edges.append((i, j))
    
    G = nx.MultiGraph()
    G.add_edges_from(edges)
    
    routes = []
    if 0 in G:
        for nbr in list(G.neighbors(0)):
            if G.has_edge(0, nbr):
                # trace route
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
    
    sol = Solution(routes=routes)
    sol.evaluate(instance)
    return sol

def solve(
    instance,
    warm_start: 'Solution' = None,
    time_limit_s: float = 3600.0,
    verbose: bool = False,
    cuts: list = None
) -> dict:
    if cuts is None:
        cuts = ['route']
        
    model, x_vars, theta_vars = build_master(instance)
    
    cb = SolverCallback(instance, x_vars, theta_vars, cuts)
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_limit_s
    model.Params.Threads = 1
    if not verbose:
        model.Params.OutputFlag = 0
    
    model.optimize(cb.callback)
    
    result = {
        'objective': float('inf'),
        'gap': model.MIPGap * 100 if model.SolCount > 0 else 100.0,
        'n_nodes': int(model.NodeCount),
        'solve_time_s': model.Runtime,
        'cuts_added': cb.cut_counts,
        'proved_optimal': model.Status == GRB.OPTIMAL
    }
    if model.SolCount > 0:
        result['solution'] = _extract_solution(model, x_vars, instance)
        result['objective'] = result['solution'].objective
    else:
        result['solution'] = None
    return result
