import gurobipy as gp

def add_jensen_set_cut(model, S, k_tilde, L_md, x_vars, theta_vars):
    lhs = gp.quicksum(theta_vars[c] for c in S)
    
    x_S = gp.quicksum(
        x_vars[min(i,j), max(i,j)]
        for i in S for j in S if i < j
    )
    
    activation = 1 + x_S - len(S) + k_tilde
    model.cbLazy(lhs >= L_md * activation)
