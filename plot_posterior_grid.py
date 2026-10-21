"""
2x5 grid of FMPE vs reference posterior samples (first two parameters) for all
10 sbibm observations, titled with the C2ST scores from the run's c2st.csv.

Run from the project root:
    python plot_posterior_grid.py --train_dir runs/lotka_volterra_full_run
"""

import argparse
import csv
from os.path import isfile, join

import matplotlib.pyplot as plt
import torch
import yaml

import sbibm
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from plot_lv_predictive import load_standardization, sample_fmpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--marker_size", type=float, default=4.0)
    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml")) as f:
        settings = yaml.safe_load(f)
    task = sbibm.get_task(settings["task"]["name"])
    standardization = load_standardization(args.train_dir, settings)
    model = build_model_from_kwargs(filename=join(args.train_dir, "best_model.pt"), device="cpu")

    c2st_path = join(args.train_dir, "c2st.csv")
    c2st = next(csv.DictReader(open(c2st_path))) if isfile(c2st_path) else {}
    labels = task.get_labels_parameters()

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for obs, ax in zip(range(1, 11), axes.flat):
        observation = task.get_observation(num_observation=obs)
        reference = task.get_reference_posterior_samples(num_observation=obs)
        reference = reference[torch.randperm(len(reference))[:args.n_samples]]
        fmpe = sample_fmpe(model, task, observation, standardization, args.n_samples)
        print(f"obs {obs}: {len(fmpe)} FMPE samples")

        ax.scatter(fmpe[:, 0], fmpe[:, 1], s=args.marker_size, alpha=0.3,
                   color="tab:blue", label="FMPE", rasterized=True)
        ax.scatter(reference[:, 0], reference[:, 1], s=args.marker_size, alpha=0.3,
                   color="tab:orange", label="reference", rasterized=True)
        score = c2st.get(f"C2ST {obs}")
        ax.set_title(f"obs {obs}" + (f" — C2ST {float(score):.3f}" if score else ""))
        ax.set(xlabel=labels[0], ylabel=labels[1])
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(markerscale=4 / args.marker_size * 3, fontsize=9)
    fig.suptitle(f"{task.name_display}: FMPE vs reference posterior "
                 f"({labels[0]} vs {labels[1]})", fontsize=14)
    fig.tight_layout()
    out = join(args.train_dir, "posteriors_grid.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
