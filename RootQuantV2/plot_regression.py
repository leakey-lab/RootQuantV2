"""plot_regression.py — regression scatter (pred vs true) from an infer.py CSV.

Reads the CSV written by ``infer.py`` (columns: image, pred_length, pred_area,
true_length, true_area; raw units, predictions already z-score-inverted and
clamped >=0) and renders a two-panel pred-vs-true regression figure for
``length_mm`` and ``area_mm2``.

Present rows only — the same mask infer.py reports on: true_length>0 AND
true_area>0. Each panel shows the identity line (y=x), an OLS best-fit line, and
R2 / RMSE / MAE / Pearson-r / n annotations. R2 matches infer.py's definition
(1 - SS_res/SS_tot about the true mean).

Usage (from the repository root; predictions.csv from ``infer.py --output_csv``):
    python -m RootQuantV2.plot_regression \
        --pred_csv predictions.csv \
        --outdir   RootQuantV2/viz_analysis/regression \
        --title    "RootQuant-V2 (D4 TTA) — test split"
"""

import argparse
import csv
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _r2(pred, true):
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def load_csv(path):
    pl, pa, tl, ta = [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pl.append(float(row["pred_length"]))
            pa.append(float(row["pred_area"]))
            tl.append(float(row["true_length"]))
            ta.append(float(row["true_area"]))
    return (np.array(pl), np.array(pa), np.array(tl), np.array(ta))


def panel(ax, true, pred, name, unit):
    n = len(true)
    err = pred - true
    r2 = _r2(pred, true)
    rmse = math.sqrt(float(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r = float(np.corrcoef(true, pred)[0, 1]) if n > 1 else float("nan")

    lo = 0.0
    hi = float(max(true.max(), pred.max())) * 1.05

    # density scatter so overplotted regions read correctly
    ax.scatter(true, pred, s=10, alpha=0.30, edgecolors="none",
               color="#1f77b4", rasterized=True)

    # identity y = x
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal (y = x)")

    # OLS best fit pred = m*true + b
    if n > 1:
        m, b = np.polyfit(true, pred, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, m * xs + b, color="#d62728", lw=1.6,
                label=f"fit: y = {m:.3f}x + {b:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true {name} ({unit})")
    ax.set_ylabel(f"predicted {name} ({unit})")
    ax.set_title(f"{name}")

    txt = (f"$R^2$ = {r2:.4f}\n"
           f"Pearson r = {r:.4f}\n"
           f"RMSE = {rmse:.3f} {unit}\n"
           f"MAE = {mae:.3f} {unit}\n"
           f"n = {n}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7",
                                  alpha=0.9))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)
    return dict(name=name, n=n, r2=r2, rmse=rmse, mae=mae, pearson=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="RootQuantV2 — pred vs true (present rows)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pl, pa, tl, ta = load_csv(args.pred_csv)

    n_total = len(tl)
    mask = (tl > 0) & (ta > 0)
    tl_p, ta_p, pl_p, pa_p = tl[mask], ta[mask], pl[mask], pa[mask]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    s_len = panel(axes[0], tl_p, pl_p, "length", "mm")
    s_area = panel(axes[1], ta_p, pa_p, "area", "mm$^2$")

    fig.suptitle(f"{args.title}\n"
                 f"n_total = {n_total}   n_present = {int(mask.sum())} "
                 f"(true length > 0 and area > 0)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png = os.path.join(args.outdir, "regression_analysis.png")
    pdf = os.path.join(args.outdir, "regression_analysis.pdf")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"n_total={n_total}  n_present={int(mask.sum())}")
    for s in (s_len, s_area):
        print(f"  {s['name']:<7} R2={s['r2']:.4f}  RMSE={s['rmse']:.3f}  "
              f"MAE={s['mae']:.3f}  r={s['pearson']:.4f}  n={s['n']}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
