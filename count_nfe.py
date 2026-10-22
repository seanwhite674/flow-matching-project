"""
Count ODE vector-field evaluations (NFE) used to draw posterior samples, mirroring
run_sbibm.evaluate_model: one batch of 2 x 10,000 samples per sbibm observation.

Run from the project root:
    python count_nfe.py --train_dirs runs/two_moons runs/slcp runs/lotka_volterra_full_run
"""

import argparse
import csv
from os.path import join

import numpy as np
import sbibm
import yaml
from dingo.core.posterior_models.build_model import build_model_from_kwargs

from plot_lv_predictive import load_standardization


def count_nfe(train_dir):
    with open(join(train_dir, "settings.yaml")) as f:
        settings = yaml.safe_load(f)
    task = sbibm.get_task(settings["task"]["name"])
    standardization = load_standardization(train_dir, settings)
    model = build_model_from_kwargs(filename=join(train_dir, "best_model.pt"), device="cpu")

    calls = {"n": 0}
    evaluate = model.evaluate_vectorfield

    def counted(*args, **kwargs):
        calls["n"] += 1
        return evaluate(*args, **kwargs)

    model.evaluate_vectorfield = counted

    st_x = standardization["x"]
    nfe = []
    for obs in range(1, 11):
        num_samples = len(task.get_reference_posterior_samples(num_observation=obs))
        x = (task.get_observation(num_observation=obs) - st_x["mean"]) / st_x["std"]
        calls["n"] = 0
        model.sample_batch(x.repeat((2 * num_samples, 1)))
        nfe.append(calls["n"])
        print(f"  {train_dir} obs {obs}: {calls['n']} evaluations")

    with open(join(train_dir, "nfe.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["num_observation", "nfe"])
        w.writerows(zip(range(1, 11), nfe))
    return nfe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    args = parser.parse_args()

    summary = {d: count_nfe(d) for d in args.train_dirs}
    print("\nrun | NFE per obs | mean | dopri5 steps (~NFE/6)")
    for d, nfe in summary.items():
        print(f"{d} | {nfe} | {np.mean(nfe):.0f} | {np.mean(nfe) / 6:.0f}")


if __name__ == "__main__":
    main()
