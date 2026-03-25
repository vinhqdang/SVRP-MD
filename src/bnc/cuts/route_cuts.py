import gurobipy as gp

def add_route_cut(model, route, Q_md, x_vars, theta_vars, instance):
    customers = route.customers
    n_cust    = len(customers)
    
    # Edges internal to route
    route_edges = [(min(customers[j], customers[j+1]), max(customers[j], customers[j+1]))
                   for j in range(n_cust - 1)]
    
    lhs = gp.quicksum(theta_vars[c] for c in customers)
    rhs_activation = (gp.quicksum(x_vars[e] for e in route_edges) - n_cust + 2)
    
    model.cbLazy(lhs >= Q_md * rhs_activation)
