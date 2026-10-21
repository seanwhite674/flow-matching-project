"""
Export sbibm's fixed test set (10 observations per task, with true parameters
and reference posterior samples) to one .npz file per task, so other methods
can be evaluated on exactly the same data without installing sbibm.

Run from the project root:
    python export_sbibm_observations.py --tasks two_moons slcp lotka_volterra

Load with:
    d = np.load("sbibm_observations/two_moons.npz")
    d["observations"]        # (10, dim_x)        x_o for obs 1..10
    d["true_parameters"]     # (10, dim_theta)    theta used to simulate x_o
    d["reference_samples"]   # (10, 10000, dim_theta)  samples from p(theta | x_o)
"""

import argparse
from pathlib import Path

import numpy as np
import sbibm


def export(task_name, out_dir):
    task = sbibm.get_task(task_name)
    obs_ids = range(1, task.num_observations + 1)
    data = {
        "observations": np.stack(
            [task.get_observation(num_observation=i).numpy()[0] for i in obs_ids]),
        "true_parameters": np.stack(
            [task.get_true_parameters(num_observation=i).numpy()[0] for i in obs_ids]),
        "reference_samples": np.stack(
            [task.get_reference_posterior_samples(num_observation=i).numpy() for i in obs_ids]),
        "observation_ids": np.array(list(obs_ids)),
        "parameter_labels": np.array(task.get_labels_parameters()),
    }
    path = out_dir / f"{task_name}.npz"
    np.savez_compressed(path, **data)
    shapes = ", ".join(f"{k} {v.shape}" for k, v in data.items() if v.ndim > 1)
    print(f"{path}: {shapes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["two_moons", "slcp", "lotka_volterra"])
    parser.add_argument("--out_dir", default="sbibm_observations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for task_name in args.tasks:
        export(task_name, out_dir)


if __name__ == "__main__":
    main()
