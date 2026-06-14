import numpy as np
import gurobipy as gp
from gurobipy import GRB

def hard_thresholding(X, y, k, steps=100, lr=1e-2):
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(steps):
        grad = -X.T @ (y - X @ beta)

        beta = beta - lr * grad

        idx = np.argsort(np.abs(beta))[-k:]
        mask = np.zeros(p)
        mask[idx] = 1
        beta = beta * mask

    return beta

def solve_mip(X, y, k, beta_init=None, time_limit=60):
    n, p = X.shape

    model = gp.Model()

    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    beta = model.addVars(p, lb=-GRB.INFINITY, name="beta")
    z = model.addVars(p, vtype=GRB.BINARY, name="z")

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    M = 10 * np.max(np.abs(beta_ols))

    obj = 0
    for i in range(n):
        expr = y[i] - gp.quicksum(X[i, j] * beta[j] for j in range(p))
        obj += expr * expr

    model.setObjective(0.5 * obj, GRB.MINIMIZE)

    model.addConstr(gp.quicksum(z[j] for j in range(p)) <= k)

    for j in range(p):
        model.addConstr(beta[j] <= M * z[j])
        model.addConstr(beta[j] >= -M * z[j])

    if beta_init is not None:
        for j in range(p):
            beta[j].Start = float(beta_init[j])
            z[j].Start = 1.0 if abs(beta_init[j]) > 1e-6 else 0.0

    model.optimize()

    beta_sol = np.array([beta[j].X for j in range(p)])
    z_sol = np.array([z[j].X for j in range(p)])

    return beta_sol, z_sol