import numpy as np
import gurobipy as gp
from gurobipy import GRB


def compute_M_upper_bound(X, y, k, time_limit=30):
    """
    Sekcja 2.3.2 z paperu: wyznaczanie M_U przez pomocnicze QP.
    Dla każdego j rozwiązuje max/min beta_j s.t. 0.5*||y-Xb||^2 <= UB.
    UB = wartość funkcji celu z OLS (lub ridge gdy p >= n).
    """
    n, p = X.shape

    # Górne ograniczenie na wartość funkcji celu (UB)
    if n > p:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        # Ridge jako surogat gdy p >= n
        lam = 1e-3
        beta_ols = np.linalg.lstsq(
            X.T @ X + lam * np.eye(p), X.T @ y, rcond=None
        )[0]

    # UB = wartość residualna OLS (luźne ograniczenie)
    UB = 0.5 * np.sum((y - X @ beta_ols) ** 2)
    # Dodaj zapas – UB musi być >= optimum best-subset
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

            # 0.5 * b^T X^T X b - (X^T y)^T b + 0.5 y^T y <= UB
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
    # Zapas bezpieczeństwa (tau = 2 jak w papierze Sekcja 2.3.3)
    return max(MU * 2.0, 1e-3)


def hard_threshold(c, k):
    """
    Operator H_k(c): zachowuje k największych (|c_i|) elementów,
    resztę zeruje. Propozycja 3 z paperu.
    """
    beta = np.zeros_like(c)
    if k == 0:
        return beta
    idx = np.argsort(np.abs(c))[::-1][:k]
    beta[idx] = c[idx]
    return beta


def discrete_first_order(X, y, k, max_iter=1000, tol=1e-6, n_restarts=50, seed=42):
    """
    Algorytm 2 z paperu: dyskretny projected gradient descent
    z hard-thresholdingiem i line search.

    Zwraca najlepsze beta (spośród n_restarts losowych startów).
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)

    XtX = X.T @ X
    Xty = X.T @ y
    # Stała Lipschitza gradientu: L = lambda_max(X^T X)
    L = float(np.linalg.eigvalsh(XtX).max())
    if L < 1e-10:
        L = 1.0

    def g(beta):
        r = y - X @ beta
        return 0.5 * float(r @ r)

    def grad_g(beta):
        return XtX @ beta - Xty

    def polish(beta):
        """Least squares na aktywnym zbiorze (polishing współczynników)."""
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
        # Losowa inicjalizacja wokół 0 (jak w papierze: min(i-1,1)*N(0,4I))
        scale = min(restart, 1) if restart > 0 else 0
        beta = scale * rng.normal(0, 2, p)
        beta = hard_threshold(beta, k)

        prev_val = g(beta)

        for it in range(max_iter):
            grad = grad_g(beta)

            # Krok gradientowy + hard thresholding
            eta = hard_threshold(beta - grad / L, k)

            # Line search: beta_new = lam*eta + (1-lam)*beta
            # Minimalizacja g wzdłuż kierunku (eta - beta)
            d = eta - beta
            Xd = X @ d
            Xb = X @ beta
            # g(beta + lam*d) = 0.5||y - Xb - lam*Xd||^2
            # dg/dlam = -Xd^T(y - Xb - lam*Xd) = 0
            # lam* = Xd^T(y - Xb) / ||Xd||^2
            denom = float(Xd @ Xd)
            if denom > 1e-14:
                lam_opt = float(Xd @ (y - Xb)) / denom
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

        # Polishing współczynników na aktywnym zbiorze
        beta = polish(beta)
        val = g(beta)

        if val < best_val:
            best_val = val
            best_beta = beta.copy()

    return best_beta, best_val


def best_subset_mio(X, y, k, time_limit=500, beta_init=None, M=None, verbose=False):
    """
    Formułowanie MIO dla best subset selection (problem 2.1/2.5 z paperu).
    Używa formułowania Big-M z SOS-like ograniczeniami.

    Parametry
    ---------
    X : ndarray (n, p)       – macierz cech (standaryzowana)
    y : ndarray (n,)         – wektor odpowiedzi
    k : int                  – docelowa liczba niezerowych współczynników
    time_limit : float       – limit czasu w sekundach
    beta_init : ndarray|None – warm start (np. z discrete_first_order)
    M : float|None           – wartość M_U; jeśli None, wyznaczana automatycznie
    verbose : bool           – czy wyświetlać logi Gurobi

    Zwraca
    ------
    beta_sol : ndarray (p,)
    z_sol    : ndarray (p,)  – wektor binarny (które cechy są aktywne)
    mio_gap  : float         – względna luka optymalnościowa (0 = optymalny)
    obj_val  : float         – wartość funkcji celu
    """
    n, p = X.shape

    # --- Standaryzacja X (paper: ||X_j||_2 = 1, mean = 0) ---
    col_means = X.mean(axis=0)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms < 1e-10] = 1.0
    X_std = (X - col_means) / col_norms

    # Standaryzacja y (zero mean)
    y_mean = y.mean()
    y_c = y - y_mean

    # --- Wyznaczanie M_U ---
    if M is None:
        if beta_init is not None:
            # Sekcja 2.3.3: tau * ||beta_init||_inf
            M = max(2.0 * np.max(np.abs(beta_init)), 1e-3)
        else:
            # Sekcja 2.3.2: przez QP (drogie, tylko gdy brak warm start)
            if p <= 200:
                M = compute_M_upper_bound(X_std, y_c, k)
            else:
                # Dla dużych p: heurystyka przez pseudo-OLS
                if n > p:
                    b_ref = np.linalg.lstsq(X_std, y_c, rcond=None)[0]
                else:
                    b_ref = X_std.T @ y_c / (np.linalg.norm(X_std) ** 2)
                M = max(5.0 * np.max(np.abs(b_ref)), 1e-3)

    # --- Precompute X^T X i X^T y (efektywna forma kwadratowa) ---
    XtX = X_std.T @ X_std
    Xty = X_std.T @ y_c
    yty = float(y_c @ y_c)

    # --- Model Gurobi ---
    model = gp.Model()
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("TimeLimit", time_limit)

    beta = model.addVars(p, lb=-GRB.INFINITY, name="beta")
    z = model.addVars(p, vtype=GRB.BINARY, name="z")

    # Funkcja celu: 0.5 * beta^T X^T X beta - (X^T y)^T beta + 0.5 y^T y
    # Zapis przez QuadExpr – O(p^2) ale bez pętli po n
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

    # Ograniczenie kardynalności
    model.addConstr(gp.quicksum(z[j] for j in range(p)) <= k, name="card")

    # Big-M: beta_j = 0 gdy z_j = 0
    for j in range(p):
        model.addConstr(beta[j] <= M * z[j], name=f"M_upper_{j}")
        model.addConstr(beta[j] >= -M * z[j], name=f"M_lower_{j}")

    # Warm start
    if beta_init is not None:
        for j in range(p):
            # Skaluj beta_init do standaryzowanej przestrzeni
            b_scaled = beta_init[j] * col_norms[j]
            beta[j].Start = float(b_scaled)
            z[j].Start = 1.0 if abs(b_scaled) > 1e-6 else 0.0

    model.optimize()

    # --- Wyniki ---
    if model.SolCount == 0:
        return np.zeros(p), np.zeros(p), 1.0, np.inf

    beta_std = np.array([beta[j].X for j in range(p)])
    z_sol = np.array([z[j].X for j in range(p)])

    # MIO gap (certyfikat suboptymalnościi)
    if model.ObjBound > -1e8 and model.ObjVal > 1e-10:
        mio_gap = abs(model.ObjVal - model.ObjBound) / (abs(model.ObjVal) + 1e-10)
    else:
        mio_gap = 0.0

    obj_val = model.ObjVal

    # Odwróć standaryzację beta
    beta_original = beta_std / col_norms

    return beta_original, z_sol, mio_gap, obj_val


def best_subset_full(X, y, k, time_limit=500, n_restarts=50,
                     run_first_order_only=False, verbose=False):
    """
    Pełny pipeline z paperu:
    1. Algorytm 2 (discrete first-order) jako warm start
    2. MIO z warm startem

    Parametry
    ---------
    run_first_order_only : bool – jeśli True, pomija MIO (tylko Alg. 2)

    Zwraca
    ------
    dict z kluczami:
        beta_fo    – wynik discrete first-order (Alg. 2)
        beta_mio   – wynik MIO (None jeśli run_first_order_only)
        mio_gap    – luka optymalnościowa MIO
        obj_fo     – wartość f.celu dla Alg. 2
        obj_mio    – wartość f.celu dla MIO
    """
    # Krok 1: discrete first-order (warm start)
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

    # Krok 2: MIO z warm startem z Alg. 2
    beta_mio, z_sol, mio_gap, obj_mio = best_subset_mio(
        X, y, k,
        time_limit=time_limit,
        beta_init=beta_fo,
        verbose=verbose
    )

    return {
        "beta_fo": beta_fo,
        "beta_mio": beta_mio,
        "z_sol": z_sol,
        "mio_gap": mio_gap,
        "obj_fo": obj_fo,
        "obj_mio": obj_mio,
    }
