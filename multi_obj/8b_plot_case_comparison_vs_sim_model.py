#!/usr/bin/env python3
"""
Same figure as 8_plot_case_comparison.py, plus a third panel: what the model
trained on the FULLY SIMULATED training set (e.g. BPI12_sim, see
9_generate_simulated_training_set.py) would have predicted for the exact same
case, prefix and recommended actions. So for one (case_study, case_id) and
method the figure has three columns:

    1. Predicted (.joblib models trained on the REAL training set)
    2. Predicted (.joblib models trained on the SIMULATED training set, --sim_case_study)
    3. Simulated (ProSiT, mean over n_sim runs -- the same "ground truth" both
       predicted panels are compared against)

Panels 1 and 3 are read straight from the evaluation table
5_result_computation.py already built for `case_study` (case_studies/<case_study>/
evaluation_tables/<method>_all_ranks.csv) -- nothing is recomputed for them.
Panel 2 does not exist in that table (it was never computed against a second
model), so it is built here: the same query instance and the same recommended
(rank -> activity, resource) pairs are re-run through --sim_case_study's
predictive_outcome_model / predictive_time_model, the same way
5_result_computation.py's predict_batch does it for the real model. This
requires --sim_case_study's models file (case_studies/<sim_case_study>/model/
catboost_model_{label,sigmoid_mm}.joblib) and --case_study's own test_data
(for the query instance) -- both already exist once 2_training_predictive_model.py
has been run for --sim_case_study.

Why this comparison matters: it isolates how much of any predicted-vs-simulated
gap comes from the SIMULATOR (the same gap 8_plot_case_comparison.py already
shows, panel 1 vs 3) versus how much comes from the PREDICTIVE MODEL being
trained on simulator-generated data instead of the real log (panel 2 vs 3,
and panel 1 vs 2).

Usage:
    python 8b_plot_case_comparison_vs_sim_model.py
    python 8b_plot_case_comparison_vs_sim_model.py --case_study BPI12 --sim_case_study BPI12_sim --case_id 208049 --method exhaustive
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.get_features import load_case_study, get_case_study_features
from utils.recommendation_functions import build_query_instances, align_query_instance_with_model, predict_outcome_proba
from utils.simulation_functions import case_id_name
from utils.pre_processing_functions import convert_dtypes_bpi12

METHODS = ["exhaustive", "nsga2"]
DEFAULT_METHOD = "exhaustive"
EVAL_TABLES_SUBDIR = "evaluation_tables"
PLOTS_SUBDIR = "plots"

# Case studies whose resource ids are numeric-looking and must be forced to string --
# otherwise a query instance built from test_data (read back as int64) mismatches the
# str categories the model's OneHotEncoder was fit on. Mirrors 5_result_computation.py.
BPI12_DTYPE_CASE_STUDIES = {"BPI12", "BPI12_sim"}

# BPI12_sim is currently the only fully-simulated training set (see
# 9_generate_simulated_training_set.py), so the comparison defaults to it.
DEFAULT_CASE_STUDY = "BPI12"
DEFAULT_SIM_CASE_STUDY = "BPI12_sim"
DEFAULT_CASE_ID = "208049"


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
    i.e. its row count in test_log.csv. None if not found."""
    count = int((test_log[case_id_name].astype(str) == str(case_id)).sum())
    return count if count > 0 else None


def predict_batch(query_instances, acts, resources, predictive_outcome_model, predictive_time_model):
    """Predicted (status, remaining_time_sigmoid_mm) for many (query_instance, act, res) triples,
    one model call per target instead of one pair per row. Mirrors
    5_result_computation.py's predict_batch exactly (duplicated here, not imported, since that
    module's name starts with a digit and isn't meant to be imported from)."""
    outcome_rows, time_rows = [], []
    for qi, act, res in zip(query_instances, acts, resources):
        o_row = align_query_instance_with_model(qi, predictive_outcome_model).iloc[0].to_dict()
        o_row["NEXT_ACTIVITY"] = act
        o_row["NEXT_RESOURCE"] = res
        outcome_rows.append(o_row)

        t_row = align_query_instance_with_model(qi, predictive_time_model).iloc[0].to_dict()
        t_row["NEXT_ACTIVITY"] = act
        t_row["NEXT_RESOURCE"] = res
        time_rows.append(t_row)

    predicted_status = predict_outcome_proba(predictive_outcome_model, pd.DataFrame(outcome_rows))
    predicted_rt_sigmoid_mm = predictive_time_model.predict(pd.DataFrame(time_rows))
    return np.asarray(predicted_status, dtype=float), np.asarray(predicted_rt_sigmoid_mm, dtype=float)


def predict_with_sim_model(
    case_id: str,
    rows: pd.DataFrame,
    query_instances_by_case: dict,
    predictive_outcome_model,
    predictive_time_model,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Re-run this case's recommended (rank -> activity, resource) pairs, plus the
    no-recommendation baseline (the case's own historical NEXT_ACTIVITY/NEXT_RESOURCE),
    through a *different* pair of predictive models (predictive_outcome_model /
    predictive_time_model, e.g. the ones trained on BPI12_sim).

    Returns (status_per_rank, remaining_time_sigmoid_mm_per_rank, baseline_status,
    baseline_remaining_time_sigmoid_mm) -- status_per_rank/remaining_time_per_rank aligned
    1:1 with rows' row order; baseline values are NaN if this case has no query instance.
    """
    qi = query_instances_by_case.get(str(case_id))
    if qi is None:
        n = len(rows)
        return np.full(n, np.nan), np.full(n, np.nan), np.nan, np.nan

    acts = rows["rec_activity"].tolist()
    # rec_resource comes back from the evaluation-table CSV as float64 (e.g. 10972.0) since
    # pandas infers a numeric dtype for a column of resource ids with no non-numeric sentinel
    # in it -- str()'d directly that would read "10972.0", which the model's OneHotEncoder (fit
    # on categories built from the raw, non-numeric-inferred log) has never seen. Round-trip
    # through int() first so it matches the "10972" the model actually knows.
    resources = [str(int(r)) if pd.notna(r) else r for r in rows["rec_resource"]]
    status_arr, rt_sigmoid_arr = predict_batch(
        [qi] * len(rows), acts, resources, predictive_outcome_model, predictive_time_model
    )

    baseline_status_arr, baseline_rt_arr = predict_batch(
        [qi], [qi["NEXT_ACTIVITY"]], [qi["NEXT_RESOURCE"]], predictive_outcome_model, predictive_time_model
    )
    return status_arr, rt_sigmoid_arr, float(baseline_status_arr[0]), float(baseline_rt_arr[0])


def plot_case_comparison_vs_sim_model(
    base_dir: Path,
    case_study: str,
    sim_case_study: str,
    case_id: str,
    methods: list[str],
    save_dir: Path | None = None,
) -> Path:
    """Build and save the 3-panel (predicted / predicted-by-sim-trained-model / simulated)
    comparison figure for one (case_study, case_id), one row per method in `methods` that has
    evaluation data for this case_id."""
    case_dir = base_dir / "case_studies" / case_study

    # 5_result_computation.py already stores predicted (real model) and simulated remaining time
    # on the sigmoid_mm scale for `case_study` -- test_data/test_log are only needed here for the
    # query instance (panel 2) and the prefix length (title).
    train_data, test_data, test_log = load_case_study(case_study)
    if case_study in BPI12_DTYPE_CASE_STUDIES:
        test_data = convert_dtypes_bpi12(test_data, "experiment")
        test_log = convert_dtypes_bpi12(test_log, "experiment")
    prefix_length = get_prefix_length(test_log, case_id)

    print(f"Loading '{sim_case_study}' models (predicted-by-sim-trained-model panel)...")
    (
        sim_predictive_outcome_model,
        sim_predictive_time_model,
        case_id_name_local,
        _activity_column_name_local,
        _resource_column_name_local,
        _continuous_features,
        _categorical_features,
        _columns_to_remove,
    ) = get_case_study_features(sim_case_study)
    query_instances_by_case = {
        str(cid): qi for cid, qi in build_query_instances(test_data, case_id_name_local).items()
    }

    method_rows = {}
    for method in methods:
        try:
            method_rows[method] = load_case_rows(case_dir, method, case_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIPPED] {method}: {e}")
    if not method_rows:
        raise ValueError(f"No evaluation data found for case_id {case_id!r} in any of {methods}.")

    n_rows = len(method_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 6 * n_rows), sharex=True, sharey=True, squeeze=False)

    for row_idx, (method, rows) in enumerate(method_rows.items()):
        ranks = rows["rank"].tolist()
        # x = outcome (maximize), y = 1 - sigmoid_mm(remaining time) (maximize) -- the objective
        # space the Pareto search reasons in, on the same sigmoid_mm scale as the time model's
        # offline MAE. Panel 1 and 3 come straight from 5_result_computation.py; panel 2 is
        # computed here against sim_case_study's models on the same (case, rank -> act/res) pairs.
        pred_x = rows["pred_status_with_rec"].to_numpy(dtype=float)
        pred_y = 1.0 - rows["pred_remaining_time_with_rec_sigmoid_mm"].to_numpy(dtype=float)
        sim_x = rows["sim_status_method_mean"].to_numpy(dtype=float)
        sim_y = 1.0 - rows["sim_remaining_time_method_mean_sigmoid_mm"].to_numpy(dtype=float)

        sim_model_status, sim_model_rt_sigmoid, sim_model_base_status, sim_model_base_rt_sigmoid = (
            predict_with_sim_model(case_id, rows, query_instances_by_case,
                                    sim_predictive_outcome_model, sim_predictive_time_model)
        )
        sim_model_x = sim_model_status
        sim_model_y = 1.0 - sim_model_rt_sigmoid

        # Baseline (no recommendation) is rank/method independent -- every row carries the same
        # value (see 5_result_computation.py), so any row's first entry is representative.
        baseline_pred_x = float(rows["pred_status_no_rec"].iloc[0])
        baseline_pred_y = 1.0 - float(rows["pred_remaining_time_no_rec_sigmoid_mm"].iloc[0])
        baseline_sim_x = float(rows["sim_status_baseline_mean"].iloc[0])
        baseline_sim_y = 1.0 - float(rows["sim_remaining_time_baseline_mean_sigmoid_mm"].iloc[0])
        baseline_sim_model_x = sim_model_base_status
        baseline_sim_model_y = 1.0 - sim_model_base_rt_sigmoid

        panels = (
            (axes[row_idx, 0], pred_x, pred_y, baseline_pred_x, baseline_pred_y,
             f"{method} -- Predicted (trained on real {case_study})"),
            (axes[row_idx, 1], sim_model_x, sim_model_y, baseline_sim_model_x, baseline_sim_model_y,
             f"{method} -- Predicted (trained on simulated {sim_case_study})"),
            (axes[row_idx, 2], sim_x, sim_y, baseline_sim_x, baseline_sim_y,
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

        # Both predicted panels are benchmarked against the same reference, the simulated
        # (ProSiT) panel -- so the two MAE boxes answer "how far off is each model" on the same
        # scale, isolating the simulator's own gap (panel 1) from the sim-trained model's
        # additional gap (panel 2).
        finite_real = np.isfinite(pred_x) & np.isfinite(pred_y) & np.isfinite(sim_x) & np.isfinite(sim_y)
        if finite_real.any():
            mae_outcome = float(np.mean(np.abs(pred_x[finite_real] - sim_x[finite_real])))
            mae_time = float(np.mean(np.abs(pred_y[finite_real] - sim_y[finite_real])))
            axes[row_idx, 2].text(
                0.03, 0.03,
                f"real-trained model vs simulated ({int(finite_real.sum())} rank(s))\n"
                f"MAE outcome = {mae_outcome:.3f}\n"
                f"MAE 1-sigmoid_mm = {mae_time:.3f}",
                transform=axes[row_idx, 2].transAxes, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
            )

        finite_sim = np.isfinite(sim_model_x) & np.isfinite(sim_model_y) & np.isfinite(sim_x) & np.isfinite(sim_y)
        if finite_sim.any():
            mae_outcome_sim = float(np.mean(np.abs(sim_model_x[finite_sim] - sim_x[finite_sim])))
            mae_time_sim = float(np.mean(np.abs(sim_model_y[finite_sim] - sim_y[finite_sim])))
            axes[row_idx, 1].text(
                0.03, 0.03,
                f"sim-trained model vs simulated ({int(finite_sim.sum())} rank(s))\n"
                f"MAE outcome = {mae_outcome_sim:.3f}\n"
                f"MAE 1-sigmoid_mm = {mae_time_sim:.3f}",
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
    fig.suptitle(f"{case_study} (models: real vs {sim_case_study}) | case {case_id} | prefix length: {prefix_label}")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # Saved under sim_case_study's own plots/ (not case_study's) -- these figures are specifically
    # about the sim-trained model, so they live alongside BPI12_sim's other outputs rather than
    # mixed into BPI12's plain predicted-vs-simulated plots from 8_plot_case_comparison.py.
    sim_case_dir = base_dir / "case_studies" / sim_case_study
    save_dir = save_dir or (sim_case_dir / EVAL_TABLES_SUBDIR / PLOTS_SUBDIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    methods_tag = "-".join(method_rows.keys())
    out_path = save_dir / f"{case_study}_{case_id}_{methods_tag}_predicted_vs_simmodel_vs_simulated.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the predicted (real model) vs predicted (sim-trained model) vs "
                    "simulated recommendation comparison for one case_id and one method (a 1x3 figure)."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                         help="Base directory containing case_studies/ (default: .)")
    parser.add_argument("--case_study", type=str, default=DEFAULT_CASE_STUDY,
                         help="The real case study whose evaluation table/test data drive the "
                             "case, its recommendations and the Predicted/Simulated panels "
                             f"(default: {DEFAULT_CASE_STUDY}).")
    parser.add_argument("--sim_case_study", type=str, default=DEFAULT_SIM_CASE_STUDY,
                         help="Case study whose .joblib models (trained on a fully simulated "
                             f"training set) drive the extra middle panel (default: {DEFAULT_SIM_CASE_STUDY}).")
    parser.add_argument("--case_id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=METHODS,
                         help="Method to plot (default: exhaustive; nsga2 is parked). Gives a single row: "
                             "predicted vs predicted-by-sim-model vs simulated.")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Where to save the figure (default: case_studies/<sim_case_study>/evaluation_tables/plots/)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None

    try:
        out_path = plot_case_comparison_vs_sim_model(
            base_dir, args.case_study, args.sim_case_study, args.case_id, [args.method], save_dir
        )
        print(f"[OK] saved {out_path}")
    except ValueError as e:
        print(f"[SKIPPED] {e}")


if __name__ == "__main__":
    main()

# Running commands:
# python 8b_plot_case_comparison_vs_sim_model.py --case_study BPI12 --sim_case_study BPI12_sim --method exhaustive --case_id 208049
# python 8b_plot_case_comparison_vs_sim_model.py --case_study BPI12 --sim_case_study BPI12_sim --method exhaustive --case_id 201373
# python 8b_plot_case_comparison_vs_sim_model.py --case_study BPI12 --sim_case_study BPI12_sim --method exhaustive --case_id 198232
