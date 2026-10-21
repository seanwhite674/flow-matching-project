"""
Corner plot of FMPE vs reference posterior samples over all parameters:
1-D marginal histograms on the diagonal, pairwise scatters below it.

Run from the project root:
    python plot_corner.py --train_dir runs/lotka_volterra_full_run --observations 1 6
"""

import argparse
import csv
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import sbibm
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from plot_lv_predictive import load_standardization, sample_fmpe

COLORS = {"FMPE": "tab:blue", "reference": "tab:orange"}


def corner(samples, labels, title, out, marker_size):
    dim = len(labels)
    fig, axes = plt.subplots(dim, dim, figsize=(2.6 * dim, 2.6 * dim))
    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                lo = min(s[:, i].min() for s in samples.values())
                hi = max(s[:, i].max() for s in samples.values())
                bins = np.linspace(lo, hi, 60)
                for name, s in samples.items():
                    ax.hist(s[:, i], bins=bins, density=True, histtype="stepfilled",
                            alpha=0.4, color=COLORS[name], label=name)
                ax.set_yticks([])
            else:
                for name, s in samples.items():
                    ax.scatter(s[:, j], s[:, i], s=marker_size, alpha=0.25,
                               color=COLORS[name], rasterized=True)
            if i == dim - 1:
                ax.set_xlabel(labels[j], fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=12)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.1, 1.0))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--observations", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--marker_size", type=float, default=3.0)
    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml")) as f:
        settings = yaml.safe_load(f)
    task = sbibm.get_task(settings["task"]["name"])
    standardization = load_standardization(args.train_dir, settings)
    model = build_model_from_kwargs(filename=join(args.train_dir, "best_model.pt"), device="cpu")
    labels = task.get_labels_parameters()
    c2st_path = join(args.train_dir, "c2st.csv")
    c2st = next(csv.DictReader(open(c2st_path))) if isfile(c2st_path) else {}

    for obs in args.observations:
        observation = task.get_observation(num_observation=obs)
        reference = task.get_reference_posterior_samples(num_observation=obs)
        reference = reference[torch.randperm(len(reference))[:args.n_samples]]
        fmpe = sample_fmpe(model, task, observation, standardization, args.n_samples)
        corner(
            {"FMPE": fmpe.numpy(), "reference": reference.numpy()},
            labels,
            title=f"{task.name_display}, obs {obs}"
            + (f", C2ST = {float(c2st[f'C2ST {obs}']):.3f}" if f"C2ST {obs}" in c2st else ""),
            out=join(args.train_dir, f"corner_obs{obs}.png"),
            marker_size=args.marker_size,
        )


if __name__ == "__main__":
    main()
