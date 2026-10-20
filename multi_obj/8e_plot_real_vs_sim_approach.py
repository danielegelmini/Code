#!/usr/bin/env python3
"""
Compares, for one case, the WHOLE pipeline run on the real log with the whole pipeline run on the
fully simulated log (1b_generate_simulated_training_set.py), in a 2x2 figure:

    row 1, --case_study (e.g. BPI12):         Predicted | Simulated
    row 2, --sim_case_study (e.g. BPI12_sim): Predicted | Simulated

Each row is an independent run of steps 3-5 on its own case study: the models trained on its own
training set build the candidates (transition system and rare-next-activity filter on that training set),
the confidence filter and the Pareto front, so the recommended actions can differ between the rows; the
recommendations are then simulated with that case study's own simulator (the Petri net and the
parameters discovered from its own log: for BPI12_sim the simulated log_BPI12_sim.xes). The test set is
the same (1b_ copies the real one), so the same case_id is the same running case in both rows.

Nothing is recomputed here: both rows are read from the evaluation tables
5_result_computation.py built for each case study (case_studies/<cs>/evaluation_tables/<method>_all_ranks.csv).
As in 8_plot_case_comparison.py: x = outcome (maximize), y = remaining time in days (minimize), one
fixed color per rank, the no-recommendation baseline as a black diamond, the time model's predictive
interval as a vertical bar on the predicted panels and, at the same level, the prediction interval of one
more simulated run (mean +- t * s * sqrt(1 + 1/n) of the n_sim runs, see 8_plot_case_comparison.py) on the
simulated panels. All four panels share both axes; the y axis starts at 0, the intervals are not clipped
(a bar without its lower end cap goes on below 0). Since the same rank can be a different action in the two rows,
each predicted panel also lists the action (activity, resource) of every rank.

Without --case_ids, the case is chosen among those with at least one recommended rank that is
simulated (and the baseline) in BOTH rows -- i.e. at least one point of the Pareto front in each
row -- preferring the most ranks in the poorer row, then the most visually separated points
(8c_find_full_data_cases.py's spread score, summed over the two rows). --list_candidates N prints
the best N.

The figures go to pareto_fronts/pareto_fronts_bpi12_vs_bpi12sim/ (see utils/pareto_fronts_paths.py);
this script's old figures of the case study there are deleted first (--keep_old to keep them).

Usage:
    python 8e_plot_real_vs_sim_approach.py
    python 8e_plot_real_vs_sim_approach.py --case_study BPI12 --sim_case_study BPI12_sim --case_ids 208028,206621
    python 8e_plot_real_vs_sim_approach.py --list_candidates 10
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
from utils.pareto_fronts_paths import BPI12_VS_BPI12SIM_DIR, clear_old_figures

METHODS = ["exhaustive", "nsga2"]
DEFAULT_METHOD = "exhaustive"
EVAL_TABLES_SUBDIR = "evaluation_tables"
# level of the simulated intervals when the evaluation table has no pred_interval_level column
DEFAULT_INTERVAL_LEVEL = 0.95

DEFAULT_CASE_STUDY = "BPI12"
DEFAULT_SIM_CASE_STUDY = "BPI12_sim"


def color_for_rank(rank: int):
    """A fixed color per rank number, the same as 8_plot_case_comparison.py."""
    return plt.get_cmap("tab10")(int(rank - 1) % 10)


def _legend_sort_key(label: str):
    if label.startswith("rank "):
        return (0, int(label.split(" ")[1]))
    if label.startswith("No recommendation"):
        return (1, 0)
    return (2, 0)


def draw_time_interval(ax, x: float, lower_days: float, upper_days: float, color) -> bool:
    """Vertical bar with end caps from lower_days to upper_days at outcome x (as 8_plot_case_comparison.py)."""
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


def load_case_runs(base_dir: Path, case_study: str, method: str, case_id: str) -> pd.DataFrame:
    """The single simulation runs of case_id from <method>_runs.csv (rank 0 = baseline), only those in which
    the recommendation was applied (as 8_plot_case_comparison.py). Empty if the table is missing."""
    runs_path = base_dir / "case_studies" / case_study / EVAL_TABLES_SUBDIR / f"{method}_runs.csv"
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


def load_eval_table(base_dir: Path, case_study: str, method: str) -> pd.DataFrame:
    table_path = base_dir / "case_studies" / case_study / EVAL_TABLES_SUBDIR / f"{method}_all_ranks.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"No evaluation table at {table_path} -- run 5_result_computation.py "
                                f"--case_studies {case_study} first.")
    return pd.read_csv(table_path, dtype={case_id_name: str})


def complete_ranks(table: pd.DataFrame) -> pd.DataFrame:
    """Per case_id: number of recommended ranks whose predicted AND simulated points are both finite
    (a point of the Pareto front that shows up in both panels), and 8c's spread score of those
    simulated points, for the cases whose baseline is finite on both sides."""
    finite = table[["pred_status_with_rec", "pred_remaining_time_with_rec",
                    "sim_status_method_mean", "sim_remaining_time_method_mean"]].notna().all(axis=1)
    base_ok = table[["pred_status_no_rec", "pred_remaining_time_no_rec",
                     "sim_status_baseline_mean", "sim_remaining_time_baseline_mean"]].notna().all(axis=1)
    usable = table[finite & base_ok]

    baseline_days = table["sim_remaining_time_baseline_mean"].to_numpy(dtype=float) / SECONDS_PER_DAY
    time_scale = float(np.nanpercentile(baseline_days, 95)) if np.isfinite(baseline_days).any() else 1.0
    time_scale = time_scale if time_scale > 0 else 1.0
    rows = []
    for cid, g in usable.groupby(case_id_name):
        sim_x = g["sim_status_method_mean"].to_numpy(dtype=float)
        sim_y = g["sim_remaining_time_method_mean"].to_numpy(dtype=float) / SECONDS_PER_DAY / time_scale
        base_x = float(g["sim_status_baseline_mean"].iloc[0])
        base_y = float(g["sim_remaining_time_baseline_mean"].iloc[0]) / SECONDS_PER_DAY / time_scale
        spread = float(np.hypot(np.ptp(sim_x), np.ptp(sim_y))) + float(np.mean(np.hypot(sim_x - base_x, sim_y - base_y)))
        rows.append({case_id_name: str(cid), "n_ranks": len(g), "spread_score": spread})
    return pd.DataFrame(rows, columns=[case_id_name, "n_ranks", "spread_score"])


def rank_candidate_cases(real_table: pd.DataFrame, sim_table: pd.DataFrame) -> pd.DataFrame:
    """case_ids with at least one complete rank in both tables, best first: most complete ranks in the
    poorer row, then the largest summed spread score."""
    both = complete_ranks(real_table).merge(complete_ranks(sim_table), on=case_id_name, suffixes=("_real", "_sim"))
    both["min_ranks"] = both[["n_ranks_real", "n_ranks_sim"]].min(axis=1)
    both["spread_score"] = both["spread_score_real"] + both["spread_score_sim"]
    return both.sort_values(["min_ranks", "spread_score"], ascending=False).reset_index(drop=True)


def plot_row(axes_row, rows: pd.DataFrame, runs: pd.DataFrame, case_study: str, row_label: str) -> tuple[set, set]:
    """Predicted | Simulated panels of one case study for one case, each with its time intervals (runs: the
    case's single simulation runs, from load_case_runs). Returns the levels of the predicted and of the
    simulated intervals drawn (for the legend)."""
    ranks = rows["rank"].tolist()
    pred_x = rows["pred_status_with_rec"].to_numpy(dtype=float)
    pred_y = rows["pred_remaining_time_with_rec"].to_numpy(dtype=float) / SECONDS_PER_DAY
    sim_x = rows["sim_status_method_mean"].to_numpy(dtype=float)
    sim_y = rows["sim_remaining_time_method_mean"].to_numpy(dtype=float) / SECONDS_PER_DAY

    # baseline (no recommendation): the same value on every row of the case (see 5_result_computation.py)
    baseline_pred_x = float(rows["pred_status_no_rec"].iloc[0])
    baseline_pred_y = float(rows["pred_remaining_time_no_rec"].iloc[0]) / SECONDS_PER_DAY
    baseline_sim_x = float(rows["sim_status_baseline_mean"].iloc[0])
    baseline_sim_y = float(rows["sim_remaining_time_baseline_mean"].iloc[0]) / SECONDS_PER_DAY

    pred_lo, pred_hi = interval_columns(rows, "pred_remaining_time_with_rec")
    base_lo, base_hi = interval_columns(rows, "pred_remaining_time_no_rec")
    levels = set(rows["pred_interval_level"].dropna().tolist()) if "pred_interval_level" in rows.columns else set()
    # on the simulated panel, the prediction interval of one more run from the n_sim runs, same level
    sim_level = min(levels) if levels else DEFAULT_INTERVAL_LEVEL
    sim_intervals = simulated_intervals(runs, ranks, sim_level)
    sim_levels = {sim_level} if np.isfinite(sim_intervals[0]).any() or np.isfinite(sim_intervals[2]) else set()

    panels = (
        (axes_row[0], pred_x, pred_y, baseline_pred_x, baseline_pred_y,
         (pred_lo, pred_hi, float(base_lo[0]), float(base_hi[0])),
         f"{row_label} -- Predicted (models trained on {case_study})"),
        (axes_row[1], sim_x, sim_y, baseline_sim_x, baseline_sim_y, sim_intervals,
         f"{row_label} -- Simulated (simulator discovered from {case_study}, mean over n_sim runs)"),
    )
    for ax, xs, ys, base_x, base_y, intervals, title in panels:
        for i, (rank, x, y) in enumerate(zip(ranks, xs, ys)):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue  # e.g. a rank with no usable "recommendation applied" simulation run
            if intervals is not None:
                draw_time_interval(ax, x, intervals[0][i], intervals[1][i], color_for_rank(rank))
            ax.scatter(x, y, color=color_for_rank(rank), s=120, edgecolors="black", linewidths=0.6,
                       label=f"rank {rank}", zorder=5)
        if np.isfinite(base_x) and np.isfinite(base_y):
            if intervals is not None:
                draw_time_interval(ax, base_x, intervals[2], intervals[3], "black")
            ax.scatter(base_x, base_y, color="black", marker="D", s=110,
                       label="No recommendation (baseline)", zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Remaining time (days, minimize)")
        ax.set_xlabel("Outcome (maximize)")
        ax.tick_params(axis="y", labelleft=True)  # sharey hides these by default on non-first columns
        ax.tick_params(axis="x", labelbottom=True)  # sharex hides these by default on non-last rows
        ax.grid(True, linestyle=":", alpha=0.6)

    # the two rows can recommend different actions at the same rank: list them, so a color is not read as an action
    # BPI12's numeric resource ids come back from the CSV as floats (10972.0)
    resources = [str(int(r)) if isinstance(r, float) and r.is_integer() else str(r) for r in rows["rec_resource"]]
    actions = "\n".join(f"rank {rank}: {act} ({res})" for rank, act, res in zip(ranks, rows["rec_activity"], resources))
    axes_row[0].text(0.98, 0.98, actions, transform=axes_row[0].transAxes, fontsize=8, va="top", ha="right",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"))
    return levels, sim_levels


def plot_real_vs_sim_approach(base_dir: Path, case_study: str, sim_case_study: str, case_id: str,
                              method: str, real_table: pd.DataFrame, sim_table: pd.DataFrame,
                              save_dir: Path | None = None) -> Path:
    """Build and save the 2x2 figure (rows: case_study, sim_case_study; columns: predicted, simulated)."""
    row_tables = []
    for cs, table in ((case_study, real_table), (sim_case_study, sim_table)):
        rows = table[table[case_id_name] == str(case_id)].sort_values("rank").reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"case_id {case_id!r} has no recommendation in the {cs} evaluation table.")
        row_tables.append((cs, rows))

    _, _, test_log = load_case_study(case_study)
    prefix_length = int((test_log[case_id_name].astype(str) == str(case_id)).sum())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    interval_levels, sim_interval_levels = set(), set()
    for row_idx, (cs, rows) in enumerate(row_tables):
        row_label = "real log" if row_idx == 0 else "simulated log"
        runs = load_case_runs(base_dir, cs, method, case_id)
        pred_levels, sim_levels = plot_row(axes[row_idx], rows, runs, cs, f"{cs} ({row_label})")
        interval_levels |= pred_levels
        sim_interval_levels |= sim_levels

    # y axis cut at 0 after all the panels are drawn (shared, so the top still fits every panel); the
    # intervals are not clipped: a bar going below 0 runs off the bottom edge without its end cap
    for ax in axes.flat:
        ax.set_ylim(bottom=0)

    handles_by_label = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    for levels, what in ((interval_levels, "predictive interval of the remaining time (model)"),
                         (sim_interval_levels, "prediction interval of a simulated run's remaining time (from the n_sim runs)")):
        if levels:
            level_txt = "/".join(f"{lvl:.0%}" for lvl in sorted(levels))
            handles_by_label[f"{level_txt} {what}"] = Line2D(
                [0], [0], color="gray", linewidth=1.6, marker="_", markersize=14, markeredgewidth=1.6)
    ordered_labels = sorted(handles_by_label, key=_legend_sort_key)
    fig.legend([handles_by_label[label] for label in ordered_labels], ordered_labels,
               loc="lower center", ncol=len(ordered_labels), bbox_to_anchor=(0.5, -0.01))

    prefix_label = f"{prefix_length} event(s)" if prefix_length > 0 else "unknown"
    fig.suptitle(f"{case_study} vs {sim_case_study} pipeline | case {case_id} | prefix length: {prefix_label} | {method}")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    save_dir = save_dir or BPI12_VS_BPI12SIM_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{case_study}_{case_id}_{method}_real_vs_sim_approach.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="2x2 comparison of the whole pipeline on the real log (top) and on the simulated log "
                    "(bottom): predicted vs simulated, for one case.")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--sim_case_study", type=str, default=DEFAULT_SIM_CASE_STUDY)
    parser.add_argument("--case_ids", type=str, default=None,
                        help="Comma-separated cases to plot, one figure each "
                             "(default: the best candidate, see the module docstring).")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=METHODS)
    parser.add_argument("--list_candidates", type=int, default=0,
                        help="Print the best N candidate case_ids (complete ranks in both rows) and exit.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save the figures (default: pareto_fronts/pareto_fronts_bpi12_vs_bpi12sim/)")
    parser.add_argument("--keep_old", action="store_true",
                        help="Do not delete this script's old figures of the case study in the save folder first "
                             "(other figures there are never touched)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    real_table = load_eval_table(base_dir, args.case_study, args.method)
    sim_table = load_eval_table(base_dir, args.sim_case_study, args.method)
    candidates = rank_candidate_cases(real_table, sim_table)
    print(f"{len(candidates)} case_id(s) have at least one recommended rank simulated (and the baseline) "
          f"in both {args.case_study} and {args.sim_case_study}.")
    if args.list_candidates:
        print(candidates.head(args.list_candidates).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return
    if candidates.empty and args.case_ids is None:
        raise SystemExit("No case_id to plot.")

    case_ids = ([c.strip() for c in args.case_ids.split(",") if c.strip()] if args.case_ids
                else [candidates[case_id_name].iloc[0]])
    save_dir = Path(args.save_dir) if args.save_dir else BPI12_VS_BPI12SIM_DIR
    if not args.keep_old:
        clear_old_figures(save_dir, f"{args.case_study}_*_{args.method}_real_vs_sim_approach.png")
    for case_id in case_ids:
        try:
            out_path = plot_real_vs_sim_approach(base_dir, args.case_study, args.sim_case_study, case_id,
                                                 args.method, real_table, sim_table, save_dir)
            print(f"[OK] saved {out_path}")
        except ValueError as e:
            print(f"[SKIPPED] {e}")


if __name__ == "__main__":
    main()
