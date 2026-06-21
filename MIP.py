import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time


def compute_M_upper_bound(X, y, k, time_limit=30):

    n, p = X.shape

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    
    UB = 0.5 * np.sum((y - X @ beta_ols) ** 2)
    UB = max(UB, 0.5 * np.sum(y ** 2))

    M_vals = np.zeros(p)

    XtX = X.T @ X
    Xty = X.T @ y
    yty = float(y @ y)

    for j in range(p):
        best = 0.0
        for sign in [1.0, -1.0]:
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("TimeLimit", time_limit / p + 1)
            b = m.addVars(p, lb=-GRB.INFINITY, name="b")

            quad_obj = gp.QuadExpr()
            for r in range(p):
                for c in range(p):
                    if XtX[r, c] != 0:
                        quad_obj += 0.5 * XtX[r, c] * b[r] * b[c]
            lin_part = gp.LinExpr()
            for r in range(p):
                lin_part += -Xty[r] * b[r]
            m.addQConstr(quad_obj + lin_part + 0.5 * yty <= UB)

            m.setObjective(sign * b[j], GRB.MINIMIZE)
            m.optimize()

            if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                best = max(best, abs(m.ObjVal))

        M_vals[j] = best

    MU = np.max(M_vals)
    return max(MU * 2.0, 1e-3)


def hard_threshold(c, k):

    beta = np.zeros_like(c)
    idx = np.argsort(np.abs(c))[::-1][:k]
    beta[idx] = c[idx]

    return beta


def discrete_first_order(X, y, k, max_iter=1000, tol=1e-6, n_restarts=50, seed=10):

    n, p = X.shape
    rng = np.random.default_rng(seed)

    XtX = X.T @ X
    Xty = X.T @ y

    L = float(np.linalg.eigvalsh(XtX).max())
    if L < 1e-10:
        L = 1.0

    def g(beta):
        r = y - X @ beta
        return 0.5 * float(r @ r)

    def grad_g(beta):
        return XtX @ beta - Xty

    def polish(beta):

        supp = np.where(np.abs(beta) > 1e-10)[0]

        if len(supp) == 0:
            return beta
        
        beta_pol = np.zeros(p)
        Xi = X[:, supp]
        res = np.linalg.lstsq(Xi, y, rcond=None)
        beta_pol[supp] = res[0]
        return beta_pol

    best_val = np.inf
    best_beta = np.zeros(p)

    for restart in range(n_restarts):

        scale = min(restart, 1) if restart > 0 else 0
        beta = scale * rng.normal(0, 2, p)
        beta = hard_threshold(beta, k)

        prev_val = g(beta)

        for it in range(max_iter):
            grad = grad_g(beta)

            eta = hard_threshold(beta - grad / L, k)

            d = eta - beta
            Xd = X @ d
            Xb = X @ beta

            denominator = float(Xd @ Xd)
            if denominator > 1e-14:
                lam_opt = float(Xd @ (y - Xb)) / denominator
                lam_opt = np.clip(lam_opt, 0.0, 1.0)
            else:
                lam_opt = 1.0

            beta_new = lam_opt * eta + (1 - lam_opt) * beta
            beta_new = hard_threshold(beta_new, k)

            new_val = g(beta_new)

            if prev_val - new_val < tol:
                beta = beta_new
                break

            beta = beta_new
            prev_val = new_val

        beta = polish(beta)
        val = g(beta)

        if val < best_val:
            best_val = val
            best_beta = beta.copy()

    return best_beta, best_val


def best_subset_mio(X, y, k, time_limit=500, beta_init=None, M=None, trajectory_log=None):

    n, p = X.shape

    col_means = X.mean(axis=0)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms < 1e-10] = 1.0
    X_std = (X - col_means) / col_norms

    y_mean = y.mean()
    y_c = y - y_mean

    if M is None:
        if beta_init is not None:
            M = max(2.0 * np.max(np.abs(beta_init)), 1e-3)
        else:
            M = compute_M_upper_bound(X_std, y_c, k)

    XtX = X_std.T @ X_std
    Xty = X_std.T @ y_c
    yty = float(y_c @ y_c)

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    beta = model.addVars(p, lb=-GRB.INFINITY, name="beta")
    z = model.addVars(p, vtype=GRB.BINARY, name="z")

    obj = gp.QuadExpr()

    for r in range(p):
        for c in range(r, p):
            if r == c:
                obj += 0.5 * XtX[r, c] * beta[r] * beta[r]
            else:
                obj += XtX[r, c] * beta[r] * beta[c]  # symetria

    for r in range(p):
        obj += -Xty[r] * beta[r]

    obj += 0.5 * yty

    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstr(gp.quicksum(z[j] for j in range(p)) <= k, name="cardinality")

    for j in range(p):
        model.addConstr(beta[j] <= M * z[j], name=f"M_upper_{j}")
        model.addConstr(beta[j] >= -M * z[j], name=f"M_lower_{j}")

    if beta_init is not None:
        for j in range(p):
            b_scaled = beta_init[j] * col_norms[j]
            beta[j].Start = float(b_scaled)
            z[j].Start = 1.0 if abs(b_scaled) > 1e-6 else 0.0

    # trajectory log
    callback = None
    t0 = None
    if trajectory_log is not None:
        if beta_init is not None:
            b_scaled0 = beta_init * col_norms
            r0 = y_c - X_std @ b_scaled0
            trajectory_log.append((0.0, 0.5 * float(r0 @ r0)))
        t0 = time.time()

        def _log_incumbent(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_best = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj_best < 1e90:
                    trajectory_log.append((time.time() - t0, obj_best))

        callback = _log_incumbent

    if callback is not None:
        model.optimize(callback)
    else:
        model.optimize()

    if trajectory_log is not None and model.SolCount > 0:
        trajectory_log.append((time.time() - t0, model.ObjVal))

    if model.SolCount == 0:
        return np.zeros(p), np.zeros(p), 1.0, np.inf

    beta_std = np.array([beta[j].X for j in range(p)])
    z_sol = np.array([z[j].X for j in range(p)])

    if model.ObjBound > -1e8 and model.ObjVal > 1e-10:
        mio_gap = abs(model.ObjVal - model.ObjBound) / (abs(model.ObjVal) + 1e-10)
    else:
        mio_gap = 0.0

    obj_val = model.ObjVal

    beta_original = beta_std / col_norms

    return beta_original, z_sol, mio_gap, obj_val


def best_subset_full(X, y, k, time_limit=500, n_restarts=50,
                     run_first_order_only=False):

    beta_fo, obj_fo = discrete_first_order(
        X, y, k, n_restarts=n_restarts
    )

    if run_first_order_only:
        return {
            "beta_fo": beta_fo,
            "beta_mio": None,
            "mio_gap": None,
            "obj_fo": obj_fo,
            "obj_mio": None,
        }

    beta_mio, z_sol, mio_gap, obj_mio = best_subset_mio(
        X, y, k,
        time_limit=time_limit,
        beta_init=beta_fo
    )

    return {
        "beta_fo": beta_fo,
        "beta_mio": beta_mio,
        "z_sol": z_sol,
        "mio_gap": mio_gap,
        "obj_fo": obj_fo,
        "obj_mio": obj_mio,
    }
