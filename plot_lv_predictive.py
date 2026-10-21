"""
Lotka-Volterra posterior predictive trajectories for a trained FMPE model.

For each observation: sample theta from the FMPE posterior and from sbibm's
reference posterior, solve the noise-free ODE for every sample, and plot the
90% band and mean of prey u(t) and predator v(t), with the observed data.

Run from the project root:
    python plot_lv_predictive.py --train_dir runs/lotka_volterra_full_run
"""

import argparse
import sys
from os.path import isfile, join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.integrate import solve_ivp

SBI_DIR = Path(__file__).parent / "flow-matching-posterior-estimation" / "sbi-benchmark"
sys.path.insert(0, str(SBI_DIR))

import sbibm  # noqa: E402
from dingo.core.posterior_models.build_model import build_model_from_kwargs  # noqa: E402
from lotka_volterra_python import U0, T_EVAL, SUBSAMPLE, _rhs  # noqa: E402
from run_sbibm import generate_dataset  # noqa: E402


def load_standardization(train_dir, settings):
    """Stats used during training; recomputed (and cached) for older runs that didn't save them."""
    path = join(train_dir, "standardization.pt")
    if isfile(path):
        return torch.load(path)
    print("standardization.pt not found -> re-simulating the training set to recompute it "
          "(approximate: fresh simulations, not the exact training data).")
    standardization = generate_dataset(settings, batch_size=1000).standardization
    torch.save(standardization, path)
    return standardization


def trajectories(thetas):
    """Noise-free ODE solutions on the full time grid, shape (n, 2, 201); failed solves dropped."""
    out = []
    for p in np.asarray(thetas, dtype=np.float64):
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(_rhs, (0.0, 20.0), U0, t_eval=T_EVAL, args=tuple(p),
                            method="LSODA", rtol=1e-6, atol=1e-6)
        if sol.success and sol.y.shape[1] == len(T_EVAL) and np.isfinite(sol.y).all():
            out.append(sol.y)
    return np.stack(out)


def sample_fmpe(model, task, observation, standardization, n):
    st_x, st_th = standardization["x"], standardization["theta"]
    x_std = (observation - st_x["mean"]) / st_x["std"]
    theta = model.sample_batch(x_std.repeat((2 * n, 1))) * st_th["std"] + st_th["mean"]
    theta = theta[torch.isfinite(task.prior_dist.log_prob(theta))]  # drop out-of-prior samples
    return theta[:n]


def plot_band(ax, traj, color, label):
    lo, hi = np.percentile(traj, [5, 95], axis=0)
    ax.fill_between(T_EVAL, lo, hi, color=color, alpha=0.35, lw=0, label=f"{label} 90%")
    ax.plot(T_EVAL, traj.mean(axis=0), color=color, lw=1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--observations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml")) as f:
        settings = yaml.safe_load(f)
    task = sbibm.get_task("lotka_volterra")
    standardization = load_standardization(args.train_dir, settings)
    model = build_model_from_kwargs(filename=join(args.train_dir, "best_model.pt"), device="cpu")

    t_obs = T_EVAL[SUBSAMPLE]
    n_obs = len(args.observations)
    fig, axes = plt.subplots(2, n_obs, figsize=(5 * n_obs, 6.5), squeeze=False)

    for col, obs in enumerate(args.observations):
        observation = task.get_observation(num_observation=obs)
        reference = task.get_reference_posterior_samples(num_observation=obs)
        reference = reference[torch.randperm(len(reference))[:args.n_samples]]
        fmpe = sample_fmpe(model, task, observation, standardization, args.n_samples)

        traj_ref, traj_fmpe = trajectories(reference), trajectories(fmpe)
        print(f"obs {obs}: {len(traj_ref)} reference / {len(traj_fmpe)} FMPE trajectories")

        y_obs = observation.reshape(2, -1).numpy()  # [prey(10), predator(10)]
        for row, name in enumerate(["prey u", "predator v"]):
            ax = axes[row, col]
            plot_band(ax, traj_ref[:, row], "grey", "reference posterior")
            plot_band(ax, traj_fmpe[:, row], "tab:orange", "FMPE posterior")
            ax.scatter(t_obs, y_obs[row], color="tab:red", s=18, zorder=3, label="observed")
            ax.set(title=f"obs {obs}: {name}", xlabel="t")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)

    fig.suptitle("Lotka-Volterra posterior predictive trajectories (noise-free ODE)")
    fig.tight_layout()
    out = join(args.train_dir, "posterior_predictive.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
