#!/usr/bin/env python3
"""
Builds, for a single (case_study, case_id), one figure comparing the k recommended actions
(+ the no-recommendation baseline) in

    x = outcome (maximize), y = 1 - sigmoid_mm(remaining_time) (maximize)

-- the exact objective space the Pareto search itself reasons in (see
exhaustive_pareto_search / nsga2_pareto_search / select_top_k_pareto_actions in
utils/recommendation_functions.py, which run on the .joblib time model's raw sigmoid_mm
output). The figure has ONE row for the requested method (--method, default "exhaustive";
"nsga2" is parked) and two columns -- predicted (.joblib models) vs. simulated (ProSiT, mean
over n_sim runs) -- so you can see where a given rank's recommendation lands in the model's
predicted space vs. where it actually lands once simulated. So --method exhaustive gives a
single 1x2 figure (two panels), not a 2x2 grid.

Why sigmoid_mm and not raw seconds: seconds is unbounded and its scale is specific to each
case's own trace length, so it makes a poor axis. sigmoid_mm is the bounded, dataset-consistent
scale the Pareto search reasons in AND the scale the time model's offline MAE is measured on, so
plotting here in sigmoid_mm makes "predicted vs simulated" directly comparable to the offline
numbers. 5_result_computation.py stores the values already in that scale, so nothing is re-scaled
here: pred_remaining_time_*_sigmoid_mm is the time model's raw output (no seconds round-trip), and
sim_remaining_time_*_mean_sigmoid_mm is E[sigmoid_mm(RT)] averaged over the n_sim runs (each run
mapped to sigmoid_mm, then averaged -- the Monte-Carlo analogue of the per-row nonlinear map the
offline "true" sigmoid_mm applies). Both sides are measured from the same reference point, the end
of the case's replayed prefix (the decision point). Each panel is annotated with the MAE between
its predicted and simulated points over the ranks shown.

Colors are assigned per rank (a fixed color per rank number, not per case or method), and are
identical across every subplot -- and across separate runs of this script -- so the same rank
is always the same color, and a recommendation's predicted-space point and simulated-space
point are visually linkable at a glance. All subplots share the same y-axis scale (remaining
time is on the same footing everywhere), and every subplot -- predicted and simulated alike --
shows its own y tick labels.

Reads the per-case-study evaluation tables 5_result_computation.py already builds
(case_studies/<case_study>/evaluation_tables/<method>_all_ranks.csv); doesn't touch the
predictive models or simulations directly. plot_case_comparison() is the reusable unit --
intended to be called in a loop for multiple case_ids later.

Usage:
    python 8_plot_case_comparison.py
    python 8_plot_case_comparison.py --case_study BAC --case_id 201812002630 --method exhaustive
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.get_features import load_case_study
from utils.simulation_functions import case_id_name

METHODS = ["exhaustive", "nsga2"]
DEFAULT_METHOD = "exhaustive"
EVAL_TABLES_SUBDIR = "evaluation_tables"
PLOTS_SUBDIR = "plots"

DEFAULT_CASE_STUDY = "BAC"
DEFAULT_CASE_ID = "201812002630"


def color_for_rank(rank: int):
    """A fixed color per rank number (not per case_id/method), so the same rank is always the
    same color across every subplot and across separate runs of this script."""
    return plt.get_cmap("tab10")(int(rank - 1) % 10)


def load_case_rows(case_dir: Path, method: str, case_id: str) -> pd.DataFrame:
    """rank-ordered rows for one case_id from <method>_all_ranks.csv (as built by 5_result_computation.py)."""
    table_path = case_dir / EVAL_TABLES_SUBDIR / f"{method}_all_ranks.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"No evaluation table at {table_path} -- run 5_result_computation.py first.")

    df = pd.read_csv(table_path, dtype={case_id_name: str})
    rows = df[df[case_id_name] == str(case_id)].sort_values("rank").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"case_id {case_id!r} not found in {table_path}.")
    return rows


def _legend_sort_key(label: str):
    if label.startswith("rank "):
        return (0, int(label.split(" ")[1]))
    return (1, 0)


def get_prefix_length(test_log: pd.DataFrame, case_id: str) -> int | None:
    """Number of historical events already observed for case_id at the recommendation point --
    i.e. its row count in test_log.csv (the same definition 5_result_computation.py's
    build_repl_id_map uses, offset by one: repl_id = prefix_length - 1). None if not found."""
    count = int((test_log[case_id_name].astype(str) == str(case_id)).sum())
    return count if count > 0 else None


def plot_case_comparison(
    base_dir: Path,
    case_study: str,
    case_id: str,
    methods: list[str],
    save_dir: Path | None = None,
) -> Path:
    """Build and save the predicted-vs-simulated comparison figure for one (case_study, case_id),
    with one row per method in `methods` that has evaluation data for this case_id."""
    case_dir = base_dir / "case_studies" / case_study

    # 5_result_computation.py already stores predicted and simulated remaining time on the
    # sigmoid_mm scale, so nothing is re-scaled here -- test_log is only needed for the prefix length.
    _, _, test_log = load_case_study(case_study)
    prefix_length = get_prefix_length(test_log, case_id)

    method_rows = {}
    for method in methods:
        try:
            method_rows[method] = load_case_rows(case_dir, method, case_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIPPED] {method}: {e}")
    if not method_rows:
        raise ValueError(f"No evaluation data found for case_id {case_id!r} in any of {methods}.")

    n_rows = len(method_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows), sharex=True, sharey=True, squeeze=False)

    for row_idx, (method, rows) in enumerate(method_rows.items()):
        ranks = rows["rank"].tolist()
        # x = outcome (maximize), y = 1 - sigmoid_mm(remaining time) (maximize) -- the objective
        # space the Pareto search reasons in, on the same sigmoid_mm scale as the time model's
        # offline MAE. Predicted y = the model's raw sigmoid_mm output; simulated y = mean over
        # the n_sim runs of sigmoid_mm(RT_run). Both from 5_result_computation.py, no re-scaling.
        pred_x = rows["pred_status_with_rec"].to_numpy(dtype=float)
        pred_y = 1.0 - rows["pred_remaining_time_with_rec_sigmoid_mm"].to_numpy(dtype=float)
        sim_x = rows["sim_status_method_mean"].to_numpy(dtype=float)
        sim_y = 1.0 - rows["sim_remaining_time_method_mean_sigmoid_mm"].to_numpy(dtype=float)

        # Baseline (no recommendation) is rank/method independent -- every row carries the same
        # value (see 5_result_computation.py), so any row's first entry is representative.
        baseline_pred_x = float(rows["pred_status_no_rec"].iloc[0])
        baseline_pred_y = 1.0 - float(rows["pred_remaining_time_no_rec_sigmoid_mm"].iloc[0])
        baseline_sim_x = float(rows["sim_status_baseline_mean"].iloc[0])
        baseline_sim_y = 1.0 - float(rows["sim_remaining_time_baseline_mean_sigmoid_mm"].iloc[0])

        panels = (
            (axes[row_idx, 0], pred_x, pred_y, baseline_pred_x, baseline_pred_y,
             f"{method} -- Predicted (.joblib models)"),
            (axes[row_idx, 1], sim_x, sim_y, baseline_sim_x, baseline_sim_y,
             f"{method} -- Simulated (ProSiT, mean over n_sim runs)"),
        )
        for ax, xs, ys, base_x, base_y, title in panels:
            for rank, x, y in zip(ranks, xs, ys):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue  # e.g. a rank with no usable "recommendation applied" simulation run
                ax.scatter(
                    x, y, color=color_for_rank(rank), s=120, edgecolors="black", linewidths=0.6,
                    label=f"rank {rank}", zorder=5,
                )
            if np.isfinite(base_x) and np.isfinite(base_y):
                ax.scatter(
                    base_x, base_y, color="black", marker="D", s=110,
                    label="No recommendation (baseline)", zorder=5,
                )
            ax.set_title(title)
            ax.set_ylabel("1 - sigmoid_mm(remaining time)  (maximize)")
            ax.tick_params(axis="y", labelleft=True)  # sharey hides these by default on non-first columns
            ax.tick_params(axis="x", labelbottom=True)  # sharex hides these by default on non-last rows
            ax.set_xlabel("Outcome (maximize)")
            ax.grid(True, linestyle=":", alpha=0.6)

        # Predicted-vs-simulated agreement over the ranks where both sides are finite -- the same
        # kind of MAE the offline evaluation reports, but here between the model and the simulator.
        finite = np.isfinite(pred_x) & np.isfinite(pred_y) & np.isfinite(sim_x) & np.isfinite(sim_y)
        if finite.any():
            mae_outcome = float(np.mean(np.abs(pred_x[finite] - sim_x[finite])))
            mae_time = float(np.mean(np.abs(pred_y[finite] - sim_y[finite])))
            axes[row_idx, 1].text(
                0.03, 0.03,
                f"predicted vs simulated ({int(finite.sum())} rank(s))\n"
                f"MAE outcome = {mae_outcome:.3f}\n"
                f"MAE 1-sigmoid_mm = {mae_time:.3f}",
                transform=axes[row_idx, 1].transAxes, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
            )

    handles_by_label = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    ordered_labels = sorted(handles_by_label, key=_legend_sort_key)
    ordered_handles = [handles_by_label[label] for label in ordered_labels]
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=len(ordered_labels), bbox_to_anchor=(0.5, -0.02))

    prefix_label = f"{prefix_length} event(s)" if prefix_length is not None else "unknown (case_id not found in test_log.csv)"
    fig.suptitle(f"{case_study} | case {case_id} | prefix length: {prefix_label}")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    save_dir = save_dir or (case_dir / EVAL_TABLES_SUBDIR / PLOTS_SUBDIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    methods_tag = "-".join(method_rows.keys())
    out_path = save_dir / f"{case_study}_{case_id}_{methods_tag}_predicted_vs_simulated.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the predicted-vs-simulated recommendation comparison for one case_id and one method (a 1x2 figure)."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--case_id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=METHODS,
                         help="Method to plot (default: exhaustive; nsga2 is parked). Gives a single row: predicted vs simulated.")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Where to save the figure (default: case_studies/<case_study>/evaluation_tables/plots/)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None

    try:
        out_path = plot_case_comparison(base_dir, args.case_study, args.case_id, [args.method], save_dir)
        print(f"[OK] saved {out_path}")
    except ValueError as e:
        print(f"[SKIPPED] {e}")


if __name__ == "__main__":
    main()
