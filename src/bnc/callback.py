import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from src.core.route import Route
from src.oracle.route_eval import eval_route
from src.bnc.cuts.route_cuts import add_route_cut
from src.bnc.cuts.set_cuts import add_jensen_set_cut
from src.bounds.jensen import jensen_bound_set

class SolverCallback:
    def __init__(self, instance, x_vars, theta_vars, cuts_enabled):
        self.instance  = instance
        self.x_vars    = x_vars
        self.theta_vars = theta_vars
        self.cuts_enabled = cuts_enabled
        self.cut_counts = {'route': 0, 'jensen_set': 0, 'rci': 0, 'sec': 0}
    
    def callback(self, model, where):
        self.model = model
        self.where = where
        if where == GRB.Callback.MIPSOL:
            x_val = {k: model.cbGetSolution(v) for k, v in self.x_vars.items()}
            theta_val = {k: model.cbGetSolution(v) for k, v in self.theta_vars.items()}
            
            routes, subtours = self._extract_routes_and_subtours(x_val)
            
            # Subtour elimination
            for comp in subtours:
                self._add_sec(list(comp))
                
            # If there are subtours, don't bother with expensive recourse cuts
            if subtours:
                return
            
            self._separate_route_cuts(routes, x_val, theta_val)
            
            if 'jensen_set' in self.cuts_enabled:
                self._separate_jensen_set_cuts(routes, x_val, theta_val)
                
    def _extract_routes_and_subtours(self, x_val):
        edges = [(i, j) for (i, j), val in x_val.items() if val > 0.5]
        G = nx.MultiGraph()
        
        # add elements per degree
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

    def _separate_route_cuts(self, routes, x_val, theta_val):
        for r in routes:
            Q_md = eval_route(r, self.instance)
            lhs_val = sum(theta_val[c] for c in r.customers)
            if lhs_val < Q_md - 1e-4:
                add_route_cut(self.model, r, Q_md, self.x_vars, self.theta_vars, self.instance)
                self.cut_counts['route'] += 1

    def _separate_jensen_set_cuts(self, routes, x_val, theta_val):
        for r in routes:
            S = r.customers
            L_md, _ = jensen_bound_set(S, self.instance)
            lhs_val = sum(theta_val[c] for c in S)
            if lhs_val < L_md - 1e-4:
                add_jensen_set_cut(self.model, S, 1, L_md, self.x_vars, self.theta_vars)
                self.cut_counts['jensen_set'] += 1
