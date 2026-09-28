"""aggregate_metrics.py — read every per-model prediction CSV produced by
infer_dual.py and emit one tidy metrics table.

Output rows are (model x species x mode x target) with R2 / RMSE / MAE / n,
scored on present rows only (length>0 AND area>0), matching the training metric.

Usage (run from the repository root):
    python -m RootQuantV2.aggregate_metrics \
        --preds_dir RootQuantV2/runs/inference_eval/preds \
        --out_csv   RootQuantV2/runs/inference_eval/metrics_summary.csv
"""

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd


def _metrics(pred: np.ndarray, true: np.ndarray):
    if len(true) == 0:
        return float("nan"), float("nan"), float("nan")
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    rmse = math.sqrt(float(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return r2, rmse, mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No prediction CSVs found in {args.preds_dir}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        model = df["model"].iloc[0] if "model" in df.columns and len(df) else os.path.basename(fp)[:-4]
        for species in ("all", "maize", "soy"):
            sub = df if species == "all" else df[df["species"] == species]
            present = sub[(sub["true_length"] > 0) & (sub["true_area"] > 0)]
            n = len(present)
            for mode in ("vanilla", "tta"):
                for target in ("length", "area"):
                    pred = present[f"pred_{target}_{mode}"].to_numpy(dtype="float64")
                    true = present[f"true_{target}"].to_numpy(dtype="float64")
                    r2, rmse, mae = _metrics(pred, true)
                    rows.append({
                        "model": model, "species": species, "mode": mode,
                        "target": target, "n_present": n,
                        "R2": round(r2, 5), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
                    })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(out)} rows, {len(files)} models)")

    # Console pivot: combined R2 (length & area averaged) per model/species/mode.
    print("\n=== R2 summary (avg of length & area, present rows) ===")
    piv = (out.groupby(["model", "species", "mode"])["R2"].mean()
              .reset_index()
              .pivot_table(index=["model", "species"], columns="mode", values="R2"))
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(piv.round(4))


if __name__ == "__main__":
    main()
