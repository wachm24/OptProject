import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

from MIP import discrete_first_order, best_subset_mio

def run_relative_accuracy_experiment(X, y, k_values, time_limit=300,
                                      n_restarts=50, seed=10, verbose=True):
    """
    Returns {k: {"trajectory": [(t, obj), ...], "best_obj": float}}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    results = {}
    for k in k_values:
        if verbose:
            print(f"--- k = {k} ---")

        t_fo0 = time.time()
        beta_fo, obj_fo = discrete_first_order(X, y, k, n_restarts=n_restarts, seed=seed)
        if verbose:
            print(f"  discrete first order: {time.time() - t_fo0:.2f}s, obj={obj_fo:.4f}")

        trajectory_log = []
        beta_mio, z_sol, mio_gap, obj_val = best_subset_mio(
            X, y, k, time_limit=time_limit, beta_init=beta_fo,
            trajectory_log=trajectory_log,
        )

        best_obj = min(v for _, v in trajectory_log) if trajectory_log else obj_val
        results[k] = {"trajectory": trajectory_log, "best_obj": best_obj,
                       "beta": beta_mio, "mio_gap": mio_gap}

        if verbose:
            print(f"  MIO: final obj={obj_val:.4f}, "
                  f"{len(trajectory_log)} logged points, mio_gap={mio_gap:.4f}")

    return results


def plot_relative_accuracy(results, title="Relative Accuracy vs Time",
                            ax=None, save_path=None, xlim=None):
    """
    Step-function plot of (f_k(t) - f_k*) / f_k* over time, one curve per k.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = plt.cm.tab10.colors

    for i, (k, res) in enumerate(sorted(results.items())):
        traj = sorted(res["trajectory"], key=lambda tv: tv[0])
        f_star = res["best_obj"]

        times = [t for t, _ in traj]
        rel_acc = [(v - f_star) / f_star if f_star > 1e-12 else 0.0 for _, v in traj]

        # accuracy stays flat until the next improvement
        step_t, step_a = [], []
        for j in range(len(times)):
            step_t.append(times[j])
            step_a.append(rel_acc[j])
            if j + 1 < len(times):
                step_t.append(times[j + 1])
                step_a.append(rel_acc[j])

        color = colors[i % len(colors)]
        ax.plot(step_t, step_a, color=color, linewidth=1.8, label=f"k={k}")

        best_idx = int(np.argmin(rel_acc))
        ax.scatter([times[best_idx]], [rel_acc[best_idx]], color=color,
                   marker="D", zorder=5, s=40)

    ax.set_xlabel("Time (secs)")
    ax.set_ylabel("Relative Accuracy")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(title="Subset size", loc="upper right")
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig, ax
    return ax



def load_diabetes_expanded(n_sample=350, seed=0):
    d = load_diabetes()
    X0 = d.data
    y0 = d.target.astype(float)
    feat = d.feature_names

    n, p0 = X0.shape
    cols = [X0[:, j] for j in range(p0)]

    for j in range(p0):
        if feat[j] != "sex":
            cols.append(X0[:, j] ** 2)

    for j in range(p0):
        for l in range(j + 1, p0):
            cols.append(X0[:, j] * X0[:, l])

    X_full = np.column_stack(cols)
    assert X_full.shape[1] == 64, X_full.shape[1]

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_sample, replace=False)
    X = X_full[idx, :]
    y = y0[idx]

    # standardize to zero mean, l2-norm
    X = X - X.mean(axis=0)
    norms = np.linalg.norm(X, axis=0)
    norms[norms < 1e-10] = 1.0
    X = X / norms

    y = y - y.mean()

    return X, y


if __name__ == "__main__":
    X, y = load_diabetes_expanded(n_sample=350, seed=0)
    n, p = X.shape

    k_values = [9, 20, 31, 42]
    results = run_relative_accuracy_experiment(
        X, y, k_values=k_values, time_limit=120, n_restarts=50, seed=10
    )

    fig, ax = plot_relative_accuracy(
        results,
        title=f"Diabetes dataset (n={n}, p={p}) — Relative Accuracy vs Time",
        save_path="plots/diabetes_relative_accuracy.png",
        xlim=(-5, 120+5)
    )