"""
Conditional vs marginal flow-matching paths for a 1-D toy p_data with three
points z_1, z_2, z_3, written as an Excalidraw file.

Left:  conditional OT paths x_t = sigma_t x_0 + t z (straight, they cross).
Right: the marginal flow dx/dt = u_t(x) = E[u_t(x | z) | x_t = x], computed
       exactly for this discrete p_data and integrated (curved, never cross).

Run from the project root:
    .venv/bin/python slide_show/figures/make_conditional_vs_marginal.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import norm

from excalidraw_utils import Drawing

OUT = Path(__file__).parent / "conditional_vs_marginal.excalidraw"

Z = np.array([-1.8, 0.4, 1.7])                   # data points
COLORS = ["#1971c2", "#e03131", "#2f9e44"]
SIGMA_MIN = 0.05
N_PATHS = 8
X0 = norm.ppf((np.arange(N_PATHS) + 0.5) / N_PATHS)   # evenly spread noise samples

# canvas mapping: t in [0, 1] -> 360 px wide, x in [-3, 3] -> 300 px tall
W, H, SCALE = 360, 300, 50


def to_px(panel_x, t, x):
    return panel_x + t * W, H / 2 - x * SCALE


def sigma(t):
    return 1 - (1 - SIGMA_MIN) * t


def marginal_field(t, x):
    """u_t(x) = sum_i w_i(x, t) u_t(x | z_i), with posterior weights over the data points."""
    s = sigma(t)
    logw = -((x[:, None] - t * Z[None, :]) ** 2) / (2 * s**2)
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    u_cond = (Z[None, :] - (1 - SIGMA_MIN) * x[:, None]) / s
    return (w * u_cond).sum(axis=1)


def marginal_paths(steps=400):
    ts = np.linspace(0, 1, steps + 1)
    xs = [X0.copy()]
    for t0, t1 in zip(ts[:-1], ts[1:]):  # midpoint (RK2) integration
        x = xs[-1]
        k1 = marginal_field(t0, x)
        k2 = marginal_field((t0 + t1) / 2, x + (t1 - t0) / 2 * k1)
        xs.append(x + (t1 - t0) * k2)
    return ts, np.stack(xs)          # (steps+1,), (steps+1, N_PATHS)


def draw_panel(d, panel_x, title, subtitle):
    # axes
    d.polyline([to_px(panel_x, 0, -3.1), to_px(panel_x, 0, 3.1)], color="#868e96", width=1)
    d.polyline([to_px(panel_x, 1, -3.1), to_px(panel_x, 1, 3.1)], color="#868e96", width=1)
    d.text(panel_x, H + 12, "t = 0", size=16, color="#868e96")
    d.text(panel_x + W, H + 12, "t = 1", size=16, color="#868e96")
    d.text(panel_x + W / 2, -75, title, size=22)
    d.text(panel_x + W / 2, -45, subtitle, size=16, color="#868e96")

    # p_0 drawn sideways, left of t = 0
    grid = np.linspace(-3, 3, 16)
    d.polyline([(to_px(panel_x, 0, g)[0] - 90 * norm.pdf(g) / norm.pdf(0), to_px(panel_x, 0, g)[1])
                for g in grid], color="#495057", width=2, roughness=1)
    d.text(panel_x - 55, H / 2 + 115, "p₀ = N(0, I)", size=16)

    # data points at t = 1, with p_data bumps drawn sideways to the right
    for z, c, name in zip(Z, COLORS, ["z₁", "z₂", "z₃"]):
        px, py = to_px(panel_x, 1, z)
        bump = np.linspace(z - 0.45, z + 0.45, 9)
        d.polyline([(px + 55 * np.exp(-((b - z) ** 2) / (2 * 0.12**2)), to_px(panel_x, 1, b)[1])
                    for b in bump], color=c, width=2, roughness=1)
        d.dot(px, py, color=c)
        d.text(px + 75, py - 12, name, size=18, color=c, align="left")
    d.text(panel_x + W + 40, H / 2 + 115, "p_data", size=16)


d = Drawing(seed=1)

# ---- left: conditional paths (each noise sample goes straight to one z)
LEFT = 0
draw_panel(d, LEFT, "Conditional paths  p_t(x | z)", "straight lines to one data point — they cross")
assign = np.random.default_rng(3).integers(0, len(Z), N_PATHS)
for x0, k in zip(X0, assign):
    d.polyline([to_px(LEFT, 0, x0), to_px(LEFT, 1, Z[k])], color=COLORS[k], width=2, arrow=True)
    d.dot(*to_px(LEFT, 0, x0), r=4, color="#495057")
d.text(LEFT + W / 2, H + 45, "u_t(x | z) = (z − (1 − σ_min) x) / σ_t   (closed form)", size=16)

# ---- right: marginal flow (average over z), integrated exactly
RIGHT = 620
draw_panel(d, RIGHT, "Marginal flow  p_t(x)", "average over the data — curved, never cross")
ts, paths = marginal_paths()
idx = np.linspace(0, len(ts) - 1, 12).astype(int)       # few points -> hand-drawn look
for j in range(N_PATHS):
    k = int(np.argmin(np.abs(Z - paths[-1, j])))
    d.polyline([to_px(RIGHT, ts[i], paths[i, j]) for i in idx], color=COLORS[k], width=2, arrow=True)
    d.dot(*to_px(RIGHT, 0, X0[j]), r=4, color="#495057")
d.text(RIGHT + W / 2, H + 45, "u_t(x) = E[ u_t(x | z) | x ]   (learned by regression)", size=16)

d.save(OUT)
print(f"Wrote {OUT} ({len(d.elements)} elements)")
print("marginal endpoints:", np.round(paths[-1], 2))
