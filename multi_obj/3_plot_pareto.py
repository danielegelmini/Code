"""Visualise the Pareto search of ONE method for one case.

For method="exhaustive" (the only one used in production): scores every
valid (activity, resource) pair on two objectives -- outcome probability
(max) and 1 - predicted time (max) -- filters them by confidence against a
"no recommendation" baseline (gamma_cls/gamma_reg, the same confidence-as-KPI
filter as utils/recommendation_functions.py's exhaustive_pareto_search) --
the same query instance with NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN
(see build_no_recommendation_baseline_instances) -- and
plots a single 2D scatter with four colour-coded categories: candidates
discarded by the confidence filter, candidates confident enough but not on
the Pareto front, the Pareto front itself, and the top-k pairs selected by
p-dispersion.

For method="nsga2" (parked, not used in production): same as before this
filter existed -- three objectives (outcome, time, predictive uncertainty),
no confidence filter -- plotted as a 2D + 3D pair with confidence
(1 - normalised uncertainty) as point colour / third axis. Its "no
recommendation" point is the same NO_NEXT_TOKEN baseline.

Both methods need models trained on the no-recommendation copy of the
training set (2_training_predictive_model.py).

For a given case study this script picks one test-set case (either a case id
passed on the command line or, by default, the case whose front has the most
solutions and the widest spread), then builds the transition system and (for
"exhaustive") the pair-frequency system from the training log -- both cached
on disk per case study + window size, see utils/setup_cache; pass
--rebuild-cache to force a recompute.

Example usage:
    python 3_plot_pareto.py --case_study "BAC" --k 5
    python 3_plot_pareto.py --case_study "BAC" --k 5 --method exhaustive --gamma_cls 0.5 --gamma_reg 0.5
    python 3_plot_pareto.py --case_study "BAC" --k 5 --method nsga2
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from paretoset import paretoset
import random
import time
import tqdm

from utils.pre_processing_functions import convert_dtypes_bpi12, NO_NEXT_TOKEN
from utils.get_features import load_case_study, get_case_study_features
from utils.setup_cache import get_transition_graph
from utils.recommendation_functions import (
    act_with_res_func,
    build_query_instances,
    build_no_recommendation_baseline_instances,
    next_possible_activities,
    _to_row_df,
    nsga2_pareto_search,
    _build_valid_pairs,
    _evaluate_candidates,
    _compute_confidence_probabilities,
    _filter_by_confidence,
    predict_time_and_uncertainty,
    predict_outcome_proba,
    select_top_k_pareto_actions,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection, nsga2 branch only)

import warnings
warnings.filterwarnings("ignore")

def _default_forbidden_map():
    """Return the per-case-study map of activities that must never be recommended.

    Input:
        None.
    Output:
        dict[str, list[str]] mapping a case-study name to the list of activity
        labels that are forbidden as recommended next activities for that case
        study (e.g. activities that trivially close the case).
    """
    bpi17_forbidden = ["O_Accepted"]
    bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]
    return {
        "bpi17_before": bpi17_forbidden,
        "bpi17_after": bpi17_forbidden,
        "BPI12_not_reordered": ["O_ACCEPTED"],
        "BPI12_not_reordered_sim": ["O_ACCEPTED"],
        "BPI12": ["O_ACCEPTED"],
        "BPI12_sim": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }


# ---------------------------------------------------------------------------
# method="nsga2" plotting (unchanged from before the confidence-as-KPI filter)
# ---------------------------------------------------------------------------
def _draw_2d(ax, fig, *, all_x, all_y, all_conf, front_x, front_y, front_conf,
             top_k_x, top_k_y, baseline_x, baseline_y):
    """Left subplot: outcome vs 1 - time, confidence encoded as point color."""
    ax.scatter(all_x, all_y, c=all_conf, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.55, s=35)
    sc = ax.scatter(front_x, front_y, c=front_conf, cmap="viridis", vmin=0.0, vmax=1.0,
                    s=95, edgecolors="black", linewidths=1.4, zorder=5)
    ax.plot(front_x, front_y, color="grey", linestyle="--", alpha=0.5, zorder=4)
    ax.scatter(top_k_x, top_k_y, facecolors="none", edgecolors="crimson", marker="P",
               s=190, linewidths=1.9, zorder=8)
    ax.scatter(1.0, 1.0, color="green", marker="X", s=110, zorder=10)
    ax.scatter(baseline_x, baseline_y, color="orange", marker="D", s=110,
               edgecolors="black", linewidths=0.6, zorder=10)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence = 1 - norm(uncertainty)  (higher is better)")
    ax.set_xlabel("Predicted Outcome (Probability Maximize)")
    ax.set_ylabel("1 - Predicted Time (Maximize)")
    ax.set_title("2D view  --  confidence as point color")
    ax.grid(True, linestyle=":", alpha=0.7)

    plot_x_min = min(np.min(all_x), baseline_x)
    plot_x_max = max(np.max(all_x), baseline_x)
    plot_y_min = min(np.min(all_y), baseline_y)
    plot_y_max = max(np.max(all_y), baseline_y)
    margin_x = (plot_x_max - plot_x_min) * 0.05 if plot_x_max != plot_x_min else 0.05
    margin_y = (plot_y_max - plot_y_min) * 0.05 if plot_y_max != plot_y_min else 0.05
    ax.set_xlim(min(plot_x_min - margin_x, -0.05), max(plot_x_max + margin_x, 1.05))
    ax.set_ylim(min(plot_y_min - margin_y, -0.05), max(plot_y_max + margin_y, 1.05))


def _draw_3d(ax, *, all_x, all_y, all_conf, front_x, front_y, front_conf,
             top_k_x, top_k_y, top_k_conf, baseline_x, baseline_y, baseline_conf,
             elev, azim):
    """Right subplot: the same front with confidence on the third axis."""
    ax.scatter(all_x, all_y, all_conf, color="black", alpha=0.35, s=30)
    ax.scatter(front_x, front_y, front_conf, color="blue", s=80)
    ax.scatter(top_k_x, top_k_y, top_k_conf, facecolors="none", edgecolors="crimson",
               marker="P", s=180, linewidths=1.9)
    ax.scatter([1.0], [1.0], [1.0], color="green", marker="X", s=180)
    ax.scatter([baseline_x], [baseline_y], [baseline_conf], color="orange", marker="D", s=150)
    ax.set_xlabel("Predicted Outcome (max)")
    ax.set_ylabel("1 - Predicted Time (max)")
    ax.set_zlabel("Confidence (max)", labelpad=6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=1.15)
    ax.set_title("3D view  --  confidence as third axis")


def _run_and_plot_nsga2(
    *, case_study, target_case_id, query_instance, poss, valid_pairs, act_with_res,
    predictive_outcome_model, predictive_time_model, pop_size, n_generations,
    random_state, k, elev, azim, save_dir,
):
    """NSGA2 branch: same as before the confidence-as-KPI filter existed.
    Three objectives (outcome, time, predictive uncertainty), no confidence
    filter, baseline = the query instance with NEXT_ACTIVITY/NEXT_RESOURCE set
    to NO_NEXT_TOKEN. Plotted as a 2D + 3D pair with confidence
    (1 - normalised uncertainty) as point colour / third axis."""
    # "No recommendation" point: evaluate the models on the query_instance with
    # NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN (no next step given).
    baseline_row = _to_row_df(query_instance).copy()
    baseline_row["NEXT_ACTIVITY"] = NO_NEXT_TOKEN
    baseline_row["NEXT_RESOURCE"] = NO_NEXT_TOKEN
    baseline_outcome = float(predict_outcome_proba(predictive_outcome_model, baseline_row)[0])
    baseline_time_arr, baseline_unc_arr = predict_time_and_uncertainty(predictive_time_model, baseline_row)
    baseline_time = float(baseline_time_arr[0])
    baseline_unc = float(baseline_unc_arr[0])
    baseline_x = baseline_outcome
    baseline_y = 1.0 - baseline_time

    print("\nRunning method: NSGA2...")
    t_start = time.time()
    all_evals = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
    pareto_set = nsga2_pareto_search(
        query_instance=query_instance,
        possible_actions=poss,
        act_with_res=act_with_res,
        predictive_outcome_model=predictive_outcome_model,
        predictive_time_model=predictive_time_model,
        pop_size=pop_size,
        n_generations=n_generations,
        random_state=random_state,
    )
    elapsed_time = time.time() - t_start
    print(f"  (NSGA2 done in {elapsed_time:.4f}s)")

    if not pareto_set:
        print("Empty Pareto set for nsga2. Nothing to plot.")
        return

    all_x = all_evals[:, 0]
    all_y = 1.0 - all_evals[:, 1]
    all_unc = all_evals[:, 2]

    # Confidence = 1 - uncertainty, min-max normalised over all evaluated points
    # (baseline included): 0 = least reliable, 1 = most reliable. Only used for
    # the point color (2D) / the third axis (3D).
    unc_all = np.concatenate([all_unc, [baseline_unc]])
    u_lo, u_hi = float(np.min(unc_all)), float(np.max(unc_all))
    u_span = (u_hi - u_lo) or 1.0
    to_conf = lambda u: 1.0 - (np.asarray(u, dtype=float) - u_lo) / u_span
    all_conf = to_conf(all_unc)
    baseline_conf = float(to_conf(baseline_unc))

    front_x_raw = np.array([item[2] for item in pareto_set], dtype=float)
    front_y_raw = np.array([1.0 - item[3] for item in pareto_set], dtype=float)
    front_unc_raw = np.array([item[4] for item in pareto_set], dtype=float)

    pareto_vals = np.column_stack((front_x_raw, front_y_raw, front_unc_raw))
    is_pareto = paretoset(pareto_vals, sense=["max", "max", "min"])
    front_x = front_x_raw[is_pareto]
    front_y = front_y_raw[is_pareto]
    front_conf = to_conf(front_unc_raw[is_pareto])
    order = np.argsort(front_x)
    front_x, front_y, front_conf = front_x[order], front_y[order], front_conf[order]

    # Points selected by select_top_k_pareto_actions (p-dispersion over the 3
    # normalised objectives): a subset of the front, marked separately.
    top_k_pairs = select_top_k_pareto_actions(pareto_set, k=k)
    pair_to_obj = {(item[0], item[1]): (item[2], 1.0 - item[3], item[4]) for item in pareto_set}
    top_k_x = np.array([pair_to_obj[p][0] for p in top_k_pairs], dtype=float)
    top_k_y = np.array([pair_to_obj[p][1] for p in top_k_pairs], dtype=float)
    top_k_conf = to_conf(np.array([pair_to_obj[p][2] for p in top_k_pairs], dtype=float))

    # -------------------------------------------------------------------------
    # One figure, two subplots of the SAME front: 2D (left) + 3D (right).
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 9))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_2d(
        ax2d, fig,
        all_x=all_x, all_y=all_y, all_conf=all_conf,
        front_x=front_x, front_y=front_y, front_conf=front_conf,
        top_k_x=top_k_x, top_k_y=top_k_y,
        baseline_x=baseline_x, baseline_y=baseline_y,
    )
    _draw_3d(
        ax3d,
        all_x=all_x, all_y=all_y, all_conf=all_conf,
        front_x=front_x, front_y=front_y, front_conf=front_conf,
        top_k_x=top_k_x, top_k_y=top_k_y, top_k_conf=top_k_conf,
        baseline_x=baseline_x, baseline_y=baseline_y, baseline_conf=baseline_conf,
        elev=elev, azim=azim,
    )

    fig.suptitle(
        f"Pareto Front Analysis  |  Dataset: {case_study}  |  Case ID: {target_case_id}  |  "
        f"Method: NSGA2  ({elapsed_time:.2f} s)",
        fontsize=13, y=0.97,
    )

    # One shared legend for the whole figure, horizontal, under both subplots.
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="grey",
                      markersize=8, label="Evaluated pairs (2D: color = confidence / 3D: black)"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="blue",
                      markeredgecolor="black", markersize=9, label="Pareto front"),
        mlines.Line2D([], [], marker="P", color="none", markeredgecolor="crimson",
                      markerfacecolor="none", markersize=13, label=f"Top-{k} selected (p-dispersion)"),
        mlines.Line2D([], [], marker="X", color="none", markerfacecolor="green",
                      markersize=11, label="Ideal point (1, 1, 1)"),
        mlines.Line2D([], [], marker="D", color="none", markerfacecolor="orange",
                      markeredgecolor="black", markersize=9, label="No recommendation (baseline)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.005,
             "All axes are objectives to maximize: outcome probability, 1 - predicted time, "
             "confidence = 1 - min-max-normalized uncertainty.",
             ha="center", fontsize=8, style="italic")

    # tight_layout misbehaves with 3D axes -> manual margins; no bbox_inches="tight"
    # either (it clips the 3D z-axis label).
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.16, top=0.88, wspace=0.12)

    filename = f"pareto_{case_study}_{str(target_case_id).replace(':', '_')}_nsga2.jpg"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, format="jpg", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved to: {filepath}")


# ---------------------------------------------------------------------------
# method="exhaustive" plotting (confidence-as-KPI filter)
# ---------------------------------------------------------------------------
def _draw_2d_confidence(ax, fig, *, discarded_x, discarded_y, confident_x, confident_y,
                         front_x, front_y, top_k_x, top_k_y, baseline_x, baseline_y,
                         gamma_cls, gamma_reg):
    """Single 2D view: outcome vs 1 - time, four colour-coded categories."""
    ax.scatter(discarded_x, discarded_y, color="lightgrey", alpha=0.65, s=35, zorder=2)
    ax.scatter(confident_x, confident_y, color="#4C72B0", alpha=0.8, s=45, zorder=3)
    ax.plot(front_x, front_y, color="#DD8452", linestyle="--", alpha=0.5, zorder=4)
    ax.scatter(front_x, front_y, color="#DD8452", s=90, edgecolors="black",
               linewidths=1.2, zorder=5)
    ax.scatter(top_k_x, top_k_y, facecolors="none", edgecolors="crimson", marker="P",
               s=190, linewidths=1.9, zorder=8)
    ax.scatter(1.0, 1.0, color="green", marker="X", s=110, zorder=10)
    ax.scatter(baseline_x, baseline_y, color="black", marker="D", s=110,
               edgecolors="white", linewidths=0.8, zorder=10)
    ax.set_xlabel("Predicted Outcome (Probability, Maximize)")
    ax.set_ylabel("1 - Predicted Time (Maximize)")
    ax.set_title(
        f"Confidence-as-KPI filtered Pareto front (exhaustive)\n"
        f"gamma_cls={gamma_cls}   gamma_reg={gamma_reg}",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.7)

    all_x = np.concatenate([discarded_x, confident_x, front_x, [baseline_x]])
    all_y = np.concatenate([discarded_y, confident_y, front_y, [baseline_y]])
    plot_x_min, plot_x_max = float(np.min(all_x)), float(np.max(all_x))
    plot_y_min, plot_y_max = float(np.min(all_y)), float(np.max(all_y))
    margin_x = (plot_x_max - plot_x_min) * 0.05 if plot_x_max != plot_x_min else 0.05
    margin_y = (plot_y_max - plot_y_min) * 0.05 if plot_y_max != plot_y_min else 0.05
    ax.set_xlim(min(plot_x_min - margin_x, -0.05), max(plot_x_max + margin_x, 1.05))
    ax.set_ylim(min(plot_y_min - margin_y, -0.05), max(plot_y_max + margin_y, 1.05))


def _run_and_plot_exhaustive(
    *, case_study, target_case_id, query_instance, valid_pairs,
    baseline_row, predictive_outcome_model, predictive_time_model,
    gamma_cls, gamma_reg, k, save_dir,
):
    """Exhaustive branch: scores every valid pair on 2 objectives (outcome,
    1 - time), filters by confidence against the baseline -- the query
    instance with NEXT_ACTIVITY/NEXT_RESOURCE set to NO_NEXT_TOKEN, see
    build_no_recommendation_baseline_instances -- and plots ALL evaluated candidates in four
    colour-coded categories: discarded by the confidence filter, confident
    but not on the front, the Pareto front, and the top-k p-dispersion
    selection -- the same _compute_confidence_probabilities /
    _filter_by_confidence building blocks
    utils.recommendation_functions.exhaustive_pareto_search() uses."""
    print("\nRunning method: EXHAUSTIVE...")
    t_start = time.time()

    all_evals = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
    all_x = all_evals[:, 0]
    all_y = 1.0 - all_evals[:, 1]

    baseline_row_df = _to_row_df(baseline_row)
    baseline_x = float(predict_outcome_proba(predictive_outcome_model, baseline_row_df)[0])
    baseline_time_arr, _ = predict_time_and_uncertainty(predictive_time_model, baseline_row_df)
    baseline_y = 1.0 - float(baseline_time_arr[0])

    prob_outcome_better, prob_time_better = _compute_confidence_probabilities(
        valid_pairs, baseline_row, query_instance,
        predictive_outcome_model, predictive_time_model,
    )
    keep_mask = _filter_by_confidence(prob_outcome_better, prob_time_better, gamma_cls, gamma_reg)
    elapsed_time = time.time() - t_start
    print(f"  (EXHAUSTIVE done in {elapsed_time:.4f}s)")

    discarded_mask = ~keep_mask
    survivors_idx = np.where(keep_mask)[0]

    if survivors_idx.size == 0:
        print("No candidate passed the confidence filter (gamma_cls / gamma_reg too strict). Nothing to plot.")
        return

    survivor_x = all_x[survivors_idx]
    survivor_y = all_y[survivors_idx]

    is_front = paretoset(np.column_stack([survivor_x, survivor_y]), sense=["max", "max"])
    front_x = survivor_x[is_front]
    front_y = survivor_y[is_front]
    confident_x = survivor_x[~is_front]
    confident_y = survivor_y[~is_front]

    front_order = np.argsort(front_x)
    front_x, front_y = front_x[front_order], front_y[front_order]

    # Rebuild the (act, res, outcome, time, uncertainty, prob_outcome_better,
    # prob_time_better) tuples select_top_k_pareto_actions expects, restricted
    # to the confidence-filtered survivors -- same contract as
    # exhaustive_pareto_search()'s return value.
    survivor_tuples = [
        (valid_pairs[i][0], valid_pairs[i][1], float(all_evals[i, 0]), float(all_evals[i, 1]),
         float(all_evals[i, 2]), float(prob_outcome_better[i]), float(prob_time_better[i]))
        for i in survivors_idx
    ]
    top_k_pairs = select_top_k_pareto_actions(survivor_tuples, k=k)
    pair_to_xy = {
        (act, res): (float(all_evals[i, 0]), 1.0 - float(all_evals[i, 1]))
        for i, (act, res) in enumerate(valid_pairs)
    }
    top_k_x = np.array([pair_to_xy[p][0] for p in top_k_pairs], dtype=float)
    top_k_y = np.array([pair_to_xy[p][1] for p in top_k_pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    _draw_2d_confidence(
        ax, fig,
        discarded_x=all_x[discarded_mask], discarded_y=all_y[discarded_mask],
        confident_x=confident_x, confident_y=confident_y,
        front_x=front_x, front_y=front_y,
        top_k_x=top_k_x, top_k_y=top_k_y,
        baseline_x=baseline_x, baseline_y=baseline_y,
        gamma_cls=gamma_cls, gamma_reg=gamma_reg,
    )

    fig.suptitle(
        f"Pareto Front Analysis  |  Dataset: {case_study}  |  Case ID: {target_case_id}  |  "
        f"Method: EXHAUSTIVE  ({elapsed_time:.2f} s)",
        fontsize=12, y=0.97,
    )

    legend_handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="lightgrey",
                      markersize=8, label="Discarded (below gamma_cls / gamma_reg)"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="#4C72B0",
                      markersize=8, label="Confident, not on Pareto front"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="#DD8452",
                      markeredgecolor="black", markersize=9, label="Pareto front"),
        mlines.Line2D([], [], marker="P", color="none", markeredgecolor="crimson",
                      markerfacecolor="none", markersize=13, label=f"Top-{k} selected (p-dispersion)"),
        mlines.Line2D([], [], marker="X", color="none", markerfacecolor="green",
                      markersize=11, label="Ideal point (1, 1)"),
        mlines.Line2D([], [], marker="D", color="none", markerfacecolor="black",
                      markeredgecolor="white", markersize=9,
                      label="No recommendation (baseline)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.text(
        0.5, 0.005,
        f"{len(valid_pairs)} evaluated pairs: {int(discarded_mask.sum())} discarded, "
        f"{len(confident_x)} confident (off-front), {len(front_x)} on the Pareto front, "
        f"{len(top_k_pairs)} selected.",
        ha="center", fontsize=8, style="italic",
    )

    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.21, top=0.86)

    filename = f"pareto_{case_study}_{str(target_case_id).replace(':', '_')}_exhaustive.jpg"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, format="jpg", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved to: {filepath}")


def run_and_plot_comparison(
    case_study: str,
    target_case_id: str = None,
    window_size: int = 5,
    pop_size: int = 50,
    n_generations: int = 10,
    random_state: int = 1234,
    k: int = 5,
    method: str = "exhaustive",
    gamma_cls: float = 0.5,
    gamma_reg: float = 0.5,
    elev: float = 22.0,
    azim: float = 0,
    rebuild_cache: bool = False,
    save_dir: str = None
):
    """Run ONE Pareto search for one case and plot its result.

    For the chosen case the function evaluates every valid (activity, resource)
    pair and saves a figure to ``save_dir``. method="exhaustive" (default)
    additionally filters candidates by confidence against the no-recommendation
    baseline (see build_no_recommendation_baseline_instances) and plots a single
    2D view with four colour-coded categories (see _run_and_plot_exhaustive);
    method="nsga2" plots the original 2D + 3D pair with confidence as
    colour/third axis and no filter (see _run_and_plot_nsga2).

    Input:
        case_study: dataset name (e.g. "BAC", "BPI12", "bpi17_before").
        target_case_id: case id to analyse; if None or not present in the test
            set, the case whose Pareto front -- built on the confidence-filtered
            candidates, as in the plot (method="exhaustive") -- spans the
            largest area is selected automatically.
        window_size: prefix window length used to build the transition system
            (and, for method="exhaustive", the pair-frequency system) and to
            look up the next possible activities.
        pop_size: NSGA-II population size (only used when method="nsga2").
        n_generations: number of NSGA-II generations (only used when method="nsga2").
        random_state: seed for numpy / random and for NSGA-II reproducibility.
        k: number of Pareto points to highlight as the top-k selection.
        method: "exhaustive" (default) or "nsga2" -- the single search whose
            front is plotted.
        gamma_cls, gamma_reg: confidence-as-KPI thresholds (method="exhaustive"
            only) -- minimum required P(candidate beats baseline) on the
            outcome / time objective respectively.
        elev, azim: elevation and azimuth (degrees) of the 3D camera
            (method="nsga2" only).
        save_dir: directory where the output .jpg figure is written (created
            if missing). Defaults to case_studies/<case_study>/pareto_front_images
            so each case study's plots stay together instead of a single shared
            folder outside case_studies/.
    Output:
        None. The figure is written to disk and progress is printed to stdout.
        The function returns early (printing a message) if the case has no
        possible next activity, no valid action-resource pair, an empty
        Pareto set, or (method="exhaustive") no candidate passes the
        confidence filter.
    """
    if save_dir is None:
        save_dir = os.path.join("case_studies", case_study, "pareto_front_images")

    np.random.seed(random_state)
    random.seed(random_state)

    print(f"Loading data for {case_study}...")
    train_data, test_data, test_log = load_case_study(case_study)

    if case_study in {"BPI12_not_reordered", "BPI12_not_reordered_sim", "BPI12", "BPI12_sim"}:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")
        test_log  = convert_dtypes_bpi12(test_log, "experiment")

    print("Getting features and models...")
    (
        predictive_outcome_model,
        predictive_time_model,
        case_id_name,
        activity_column_name,
        resource_column_name,
        continuous_features,
        categorical_features,
        columns_to_remove,
    ) = get_case_study_features(case_study)

    print("Building transition system and maps...")
    transition_graph = get_transition_graph(
        case_study,
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size,
        rebuild=rebuild_cache,
    )

    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
    forbidden_map = _default_forbidden_map()
    forbidden = set(forbidden_map.get(case_study, []))

    query_instances_by_case = build_query_instances(test_data, case_id_name)
    # The "no recommendation" baseline for the confidence-as-KPI filter: the
    # SAME query instance the candidates are evaluated on, with NEXT_ACTIVITY/
    # NEXT_RESOURCE set to NO_NEXT_TOKEN (see
    # build_no_recommendation_baseline_instances). It exists for every case.
    baseline_instances_by_case = build_no_recommendation_baseline_instances(query_instances_by_case)
    unique_cases = pd.unique(test_data[case_id_name])

    # Automatic selection of the case whose Pareto front -- built, like the plot,
    # only on the candidates that pass the confidence filter -- spans the largest
    # area (outcome range x (1 - time) range); ties broken by the number of front points.
    if target_case_id is None or target_case_id not in unique_cases:
        print("Searching for the case whose confidence-filtered Pareto front spans the largest area...")
        best_score = (-1.0, -1)
        best_case_id = None

        for cid in tqdm.tqdm(unique_cases, desc="Evaluating cases"):
            trace_df = test_log[test_log[case_id_name] == cid]
            trace_history = trace_df[activity_column_name].tolist()
            query_instance = query_instances_by_case[cid]

            poss = next_possible_activities(trace_history, transition_graph, window_size)
            poss = [a for a in poss if a not in forbidden]
            if not poss:
                continue

            valid_pairs = _build_valid_pairs(poss, act_with_res)
            n_solutions = len(valid_pairs)
            if n_solutions < 10:
                continue

            all_evals = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
            all_x = all_evals[:, 0]
            all_y = 1.0 - all_evals[:, 1]

            try:
                if method == "exhaustive":
                    prob_out, prob_time = _compute_confidence_probabilities(
                        valid_pairs, baseline_instances_by_case[cid], query_instance,
                        predictive_outcome_model, predictive_time_model,
                    )
                    keep = _filter_by_confidence(prob_out, prob_time, gamma_cls, gamma_reg)
                    if not keep.any():
                        continue
                    all_x, all_y = all_x[keep], all_y[keep]

                is_pareto = paretoset(np.column_stack((all_x, all_y)), sense=["max", "max"])
                front_x = all_x[is_pareto]
                front_y = all_y[is_pareto]

                spread_area = (np.max(front_x) - np.min(front_x)) * (np.max(front_y) - np.min(front_y))
                score = (spread_area, len(front_x))

                if score > best_score:
                    best_score = score
                    best_case_id = cid
            except Exception:
                continue

        if best_case_id is None:
            target_case_id = random.choice(unique_cases)
        else:
            target_case_id = best_case_id
        print(f"\nAutomatically selected case: {target_case_id}")
    else:
        print(f"Using the provided case_id: {target_case_id}")

    # Extract data for the chosen case
    trace_df = test_log[test_log[case_id_name] == target_case_id]
    trace_history = trace_df[activity_column_name].tolist()
    query_instance = query_instances_by_case[target_case_id]

    poss = next_possible_activities(trace_history, transition_graph, window_size)
    poss = [a for a in poss if a not in forbidden]

    if not poss:
        print("No possible next activity for this case.")
        return

    valid_pairs = _build_valid_pairs(poss, act_with_res)
    if not valid_pairs:
        print("No valid action-resource pair.")
        return

    method = method.lower()
    if method not in {"exhaustive", "nsga2"}:
        raise ValueError("method must be either 'exhaustive' or 'nsga2'.")

    os.makedirs(save_dir, exist_ok=True)

    if method == "exhaustive":
        _run_and_plot_exhaustive(
            case_study=case_study, target_case_id=target_case_id,
            query_instance=query_instance,
            valid_pairs=valid_pairs, baseline_row=baseline_instances_by_case.get(target_case_id),
            predictive_outcome_model=predictive_outcome_model,
            predictive_time_model=predictive_time_model,
            gamma_cls=gamma_cls, gamma_reg=gamma_reg, k=k, save_dir=save_dir,
        )
    else:
        _run_and_plot_nsga2(
            case_study=case_study, target_case_id=target_case_id,
            query_instance=query_instance, poss=poss, valid_pairs=valid_pairs,
            act_with_res=act_with_res,
            predictive_outcome_model=predictive_outcome_model,
            predictive_time_model=predictive_time_model,
            pop_size=pop_size, n_generations=n_generations, random_state=random_state,
            k=k, elev=elev, azim=azim, save_dir=save_dir,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Pareto comparison with execution times.')
    parser.add_argument('--case_study', type=str, required=True, help='Dataset name')
    parser.add_argument('--case_id', type=str, default=None, help='Specific case ID (optional)')
    parser.add_argument('--k', type=int, default=5, help='Number of top-k points to highlight (default: 5)')
    parser.add_argument('--method', type=str, default='exhaustive', choices=['exhaustive', 'nsga2'],
                        help="Search method whose front is plotted (default: exhaustive; nsga2 is parked)")
    parser.add_argument('--gamma_cls', type=float, default=0.5,
                        help="Confidence-as-KPI threshold (method='exhaustive' only): minimum "
                             "required P(candidate outcome beats baseline) (default: 0.5)")
    parser.add_argument('--gamma_reg', type=float, default=0.5,
                        help="Confidence-as-KPI threshold (method='exhaustive' only): minimum "
                             "required P(candidate time beats baseline) (default: 0.5)")
    parser.add_argument('--elev', type=float, default=22.0,
                        help="3D camera elevation in degrees (method='nsga2' only, default: 22)")
    parser.add_argument('--azim', type=float, default=-45.0,
                        help="3D camera azimuth in degrees (method='nsga2' only, default: -45)")
    parser.add_argument('--rebuild-cache', dest='rebuild_cache', action='store_true',
                        help="Force recomputing the (cached) transition system instead of loading it")

    try:
        args = parser.parse_args()
        run_and_plot_comparison(case_study=args.case_study, target_case_id=args.case_id, k=args.k,
                                method=args.method, gamma_cls=args.gamma_cls, gamma_reg=args.gamma_reg,
                                elev=args.elev, azim=args.azim, rebuild_cache=args.rebuild_cache)
    except Exception as e:
        print(f"Error during execution: {e}")


# example usage:
# python 3_plot_pareto.py --case_study "BAC" --k 5
# python 3_plot_pareto.py --case_study "BAC" --k 5 --method exhaustive --gamma_cls 0.5 --gamma_reg 0.5
# python 3_plot_pareto.py --case_study "BAC" --k 5 --method nsga2 --elev 30 --azim 45
