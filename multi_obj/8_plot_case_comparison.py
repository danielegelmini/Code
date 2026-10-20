#!/usr/bin/env python3
"""
Builds, for a single (case_study, case_id), one figure comparing the k recommended actions
(+ the no-recommendation baseline) in

    x = outcome (maximize), y = remaining time in days (minimize)

-- the exact objective space the Pareto search itself reasons in (see
exhaustive_pareto_search / select_top_k_pareto_actions in utils/recommendation_functions.py,
which run on the .joblib time model's prediction in days).The figure has ONE row for the requested method (--method, default "exhaustive";
"nsga2" is parked) and two columns -- predicted (.joblib models) vs. simulated (ProSiT, mean
over n_sim runs) -- so you can see where a given rank's recommendation lands in the model's
predicted space vs. where it actually lands once simulated. So --method exhaustive gives a
single 1x2 figure (two panels), not a 2x2 grid.

The remaining time is in days (since 2026-10-12 the time model learns days, not sigmoid_mm), the
same unit the time model's offline MAE is measured in: pred_remaining_time_* is the model's
prediction and sim_remaining_time_*_mean the mean over the n_sim runs, both stored in seconds by
5_result_computation.py and divided by 86400 here. Both sides are measured from the same reference
point, the end of the case's replayed prefix (the decision point).

On the predicted panel every point also carries a vertical bar: the time model's predictive
interval for the remaining time (equal-tailed, level pred_interval_level, 95% by default; total
recalibrated uncertainty, see predict_time_interval), read from the
pred_remaining_time_*_{lower,upper} columns of the evaluation table. With the log1p_days time model
it is asymmetric around the point, which is the log-normal mean. Tables built before these columns
existed are plotted without bars (re-run 5_result_computation.py to get them).

On the simulated panel every point carries the same kind of bar, at the same level: the prediction
interval of the remaining time of one more simulated run, from the n_sim runs the point's mean is made
of (<method>_runs.csv, the runs in which the recommendation was applied; rank 0 for the baseline), in days:
mean +- t_{(1+level)/2, n-1} * s * sqrt(1 + 1/n), s the standard deviation of the runs' remaining times
(Student t because s is estimated from n_sim runs). The runs' times being strongly skewed, the lower end
is below 0 for most points: the y axis starts at 0 and the interval is not clipped, so a bar without its
lower end cap goes on below the bottom edge.

Colors are assigned per rank (a fixed color per rank number, not per case or method), and are
identical across every subplot -- and across separate runs of this script -- so the same rank
is always the same color, and a recommendation's predicted-space point and simulated-space
point are visually linkable at a glance. All subplots share the same y-axis scale (remaining
time is on the same footing everywhere), and every subplot -- predicted and simulated alike --
shows its own y tick labels.

Reads the per-case-study evaluation tables 5_result_computation.py already builds
(case_studies/<case_study>/evaluation_tables/<method>_all_ranks.csv); doesn't touch the
predictive models or simulations directly. plot_case_comparison() is the reusable unit,
called once per case_id of --case_ids.

The figures go to pareto_fronts/pareto_fronts_predicted_vs_simulated/<case_study>/ (see
utils/pareto_fronts_paths.py); the case study's old figures there are deleted first, so the folder
holds only the case_ids of the last run (--keep_old to keep them).

Usage:
    python 8_plot_case_comparison.py --case_study BAC --case_ids 20191002825,20191012606 --method exhaustive
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from utils.get_features import load_case_study
from utils.simulation_functions import case_id_name
from utils.predictive_models_functions import SECONDS_PER_DAY
from utils.pareto_fronts_paths import PREDICTED_VS_SIMULATED_DIR, clear_old_figures

METHODS = ["exhaustive", "nsga2"]
DEFAULT_METHOD = "exhaustive"
EVAL_TABLES_SUBDIR = "evaluation_tables"
# level of the simulated intervals when the evaluation table has no pred_interval_level column
DEFAULT_INTERVAL_LEVEL = 0.95

DEFAULT_CASE_STUDY = "BAC"


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
    if label.startswith("No recommendation"):
        return (1, 0)
    return (2, 0)


def draw_time_interval(ax, x: float, lower_days: float, upper_days: float, color) -> bool:
    """Vertical bar with end caps from lower_days to upper_days at outcome x (the predictive interval
    of the remaining time); drawn as a line, not errorbar(), since the point (log-normal mean) can fall
    outside the interval. False (nothing drawn) if the interval is not finite."""
    if not (np.isfinite(x) and np.isfinite(lower_days) and np.isfinite(upper_days)):
        return False
    ax.plot([x, x], [lower_days, upper_days], color=color, linewidth=1.6, alpha=0.8,
            marker="_", markersize=14, markeredgewidth=1.6, zorder=4)
    return True


def interval_columns(rows: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) in days of the predictive interval stored as <prefix>_lower / <prefix>_upper,
    all-NaN if the evaluation table predates these columns."""
    if f"{prefix}_lower" not in rows.columns or f"{prefix}_upper" not in rows.columns:
        nan = np.full(len(rows), np.nan)
        return nan, nan.copy()
    return (rows[f"{prefix}_lower"].to_numpy(dtype=float) / SECONDS_PER_DAY,
            rows[f"{prefix}_upper"].to_numpy(dtype=float) / SECONDS_PER_DAY)


def load_case_runs(case_dir: Path, method: str, case_id: str) -> pd.DataFrame:
    """The single simulation runs of case_id from <method>_runs.csv (rank 0 = baseline), only those in which
    the recommendation was applied -- the runs the sim_*_mean columns are the mean of. Empty if the table
    is missing (then no simulated interval is drawn)."""
    runs_path = case_dir / EVAL_TABLES_SUBDIR / f"{method}_runs.csv"
    if not runs_path.exists():
        print(f"[WARNING] no {runs_path} -- re-run 5_result_computation.py to draw the simulated intervals.")
        return pd.DataFrame(columns=["rank", "remaining_time"])
    runs = pd.read_csv(runs_path, dtype={case_id_name: str})
    return runs[(runs[case_id_name] == str(case_id)) & runs["applied"].astype(bool)]


def simulated_time_interval(times_seconds, level: float) -> tuple[float, float]:
    """(lower, upper) in days: the equal-tailed `level` prediction interval of the remaining time of one more
    simulated run, from the runs' remaining times (seconds): mean +- t_{(1+level)/2, n-1} * s * sqrt(1 + 1/n),
    in days. (nan, nan) with fewer than 2 runs."""
    days = np.asarray(times_seconds, dtype=float) / SECONDS_PER_DAY
    days = days[np.isfinite(days)]
    n = len(days)
    if n < 2:
        return np.nan, np.nan
    half_width = stats.t.ppf((1 + level) / 2, n - 1) * days.std(ddof=1) * np.sqrt(1 + 1 / n)
    return float(days.mean() - half_width), float(days.mean() + half_width)


def simulated_intervals(runs: pd.DataFrame, ranks: list, level: float) -> tuple:
    """(lower per rank, upper per rank, baseline lower, baseline upper) in days, see simulated_time_interval."""
    def interval(rank):
        return simulated_time_interval(runs.loc[runs["rank"] == rank, "remaining_time"], level)
    per_rank = [interval(rank) for rank in ranks]
    base_lo, base_hi = interval(0)
    return (np.array([lo for lo, _ in per_rank]), np.array([hi for _, hi in per_rank]), base_lo, base_hi)


def interval_legend_entry(interval_levels: set, what: str) -> tuple[str, Line2D] | None:
    """Legend (label, handle) for interval bars described by `what`, None if no bar could be drawn."""
    if not interval_levels:
        return None
    level_txt = "/".join(f"{lvl:.0%}" for lvl in sorted(interval_levels))
    handle = Line2D([0], [0], color="gray", linewidth=1.6, marker="_", markersize=14, markeredgewidth=1.6)
    return f"{level_txt} {what}", handle


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

    # 5_result_computation.py stores predicted and simulated remaining times in seconds; they are only
    # converted to days here -- test_log is only needed for the prefix length.
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
    interval_levels, sim_interval_levels = set(), set()
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows), sharex=True, sharey=True, squeeze=False)

    for row_idx, (method, rows) in enumerate(method_rows.items()):
        ranks = rows["rank"].tolist()
        # x = outcome (maximize), y = remaining time in days (minimize) -- the objective space the
        # Pareto search reasons in (since 2026-10-12, when the time model moved from sigmoid_mm to days).
        # Predicted y = the time model's prediction; simulated y = mean over the n_sim runs.
        pred_x = rows["pred_status_with_rec"].to_numpy(dtype=float)
        pred_y = rows["pred_remaining_time_with_rec"].to_numpy(dtype=float) / SECONDS_PER_DAY
        sim_x = rows["sim_status_method_mean"].to_numpy(dtype=float)
        sim_y = rows["sim_remaining_time_method_mean"].to_numpy(dtype=float) / SECONDS_PER_DAY

        # Baseline (no recommendation) is rank/method independent -- every row carries the same
        # value (see 5_result_computation.py), so any row's first entry is representative.
        baseline_pred_x = float(rows["pred_status_no_rec"].iloc[0])
        baseline_pred_y = float(rows["pred_remaining_time_no_rec"].iloc[0]) / SECONDS_PER_DAY
        baseline_sim_x = float(rows["sim_status_baseline_mean"].iloc[0])
        baseline_sim_y = float(rows["sim_remaining_time_baseline_mean"].iloc[0]) / SECONDS_PER_DAY

        # predictive interval of the remaining time on the predicted panel
        pred_lo, pred_hi = interval_columns(rows, "pred_remaining_time_with_rec")
        base_lo, base_hi = interval_columns(rows, "pred_remaining_time_no_rec")
        pred_intervals = (pred_lo, pred_hi, float(base_lo[0]), float(base_hi[0]))
        if "pred_interval_level" in rows.columns:
            interval_levels.update(rows["pred_interval_level"].dropna().tolist())
        # on the simulated panel, the prediction interval of one more run from the n_sim runs, same level
        row_levels = rows["pred_interval_level"].dropna() if "pred_interval_level" in rows.columns else []
        sim_level = float(row_levels.iloc[0]) if len(row_levels) else DEFAULT_INTERVAL_LEVEL
        sim_intervals = simulated_intervals(load_case_runs(case_dir, method, case_id), ranks, sim_level)
        if np.isfinite(sim_intervals[0]).any() or np.isfinite(sim_intervals[2]):
            sim_interval_levels.add(sim_level)

        panels = (
            (axes[row_idx, 0], pred_x, pred_y, baseline_pred_x, baseline_pred_y, pred_intervals,
             f"{method} -- Predicted (.joblib models)"),
            (axes[row_idx, 1], sim_x, sim_y, baseline_sim_x, baseline_sim_y, sim_intervals,
             f"{method} -- Simulated (ProSiT, mean over n_sim runs)"),
        )
        for ax, xs, ys, base_x, base_y, intervals, title in panels:
            for i, (rank, x, y) in enumerate(zip(ranks, xs, ys)):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue  # e.g. a rank with no usable "recommendation applied" simulation run
                if intervals is not None:
                    draw_time_interval(ax, x, intervals[0][i], intervals[1][i], color_for_rank(rank))
                ax.scatter(
                    x, y, color=color_for_rank(rank), s=120, edgecolors="black", linewidths=0.6,
                    label=f"rank {rank}", zorder=5,
                )
            if np.isfinite(base_x) and np.isfinite(base_y):
                if intervals is not None:
                    draw_time_interval(ax, base_x, intervals[2], intervals[3], "black")
                ax.scatter(
                    base_x, base_y, color="black", marker="D", s=110,
                    label="No recommendation (baseline)", zorder=5,
                )
            ax.set_title(title)
            ax.set_ylabel("Remaining time (days, minimize)")
            ax.tick_params(axis="y", labelleft=True)  # sharey hides these by default on non-first columns
            ax.tick_params(axis="x", labelbottom=True)  # sharex hides these by default on non-last rows
            ax.set_xlabel("Outcome (maximize)")
            ax.grid(True, linestyle=":", alpha=0.6)

    # y axis cut at 0 after all the panels are drawn (shared, so the top still fits every panel); the
    # intervals are not clipped: a bar going below 0 runs off the bottom edge without its end cap
    for ax in axes.flat:
        ax.set_ylim(bottom=0)

    handles_by_label = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    legend_entry = interval_legend_entry(interval_levels, "predictive interval of the remaining time (model)")
    if legend_entry is not None:
        handles_by_label[legend_entry[0]] = legend_entry[1]
    else:
        print("[WARNING] the evaluation table has no pred_remaining_time_*_lower/upper columns -- "
              "re-run 5_result_computation.py to draw the predictive intervals.")
    sim_legend_entry = interval_legend_entry(
        sim_interval_levels, "prediction interval of a simulated run's remaining time (from the n_sim runs)")
    if sim_legend_entry is not None:
        handles_by_label[sim_legend_entry[0]] = sim_legend_entry[1]
    ordered_labels = sorted(handles_by_label, key=_legend_sort_key)
    ordered_handles = [handles_by_label[label] for label in ordered_labels]
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=len(ordered_labels), bbox_to_anchor=(0.5, -0.02))

    prefix_label = f"{prefix_length} event(s)" if prefix_length is not None else "unknown (case_id not found in test_log.csv)"
    fig.suptitle(f"{case_study} | case {case_id} | prefix length: {prefix_label}")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    save_dir = save_dir or (PREDICTED_VS_SIMULATED_DIR / case_study)
    save_dir.mkdir(parents=True, exist_ok=True)
    methods_tag = "-".join(method_rows.keys())
    out_path = save_dir / f"{case_study}_{case_id}_{methods_tag}_predicted_vs_simulated.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the predicted-vs-simulated recommendation comparison (a 1x2 figure) for some case_ids "
                    "of one case study and one method."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--case_ids", type=str, required=True,
                         help="Comma-separated case_ids, one figure each (e.g. picked with 8c_find_full_data_cases.py)")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=METHODS,
                         help="Method to plot (default: exhaustive; nsga2 is parked). Gives a single row: predicted vs simulated.")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Where to save the figures (default: "
                              "pareto_fronts/pareto_fronts_predicted_vs_simulated/<case_study>/)")
    parser.add_argument("--keep_old", action="store_true",
                         help="Do not delete the case study's old figures of this method in the save folder first")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir) if args.save_dir else PREDICTED_VS_SIMULATED_DIR / args.case_study
    if not args.keep_old:
        clear_old_figures(save_dir, f"{args.case_study}_*_{args.method}_predicted_vs_simulated.png")

    for case_id in [c.strip() for c in args.case_ids.split(",") if c.strip()]:
        try:
            out_path = plot_case_comparison(base_dir, args.case_study, case_id, [args.method], save_dir)
            print(f"[OK] saved {out_path}")
        except ValueError as e:
            print(f"[SKIPPED] {e}")


if __name__ == "__main__":
    main()
