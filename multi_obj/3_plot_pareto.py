"""Visual comparison of the exhaustive and NSGA-II Pareto searches for one case.

For a given case study this script picks one test-set case (either a case id
passed on the command line or, by default, the case whose Pareto front has the
most solutions and the widest spread), then for that case:

  * builds the transition system from the training log (cached on disk per
    case study + window size -- see utils/setup_cache; pass --rebuild-cache to
    force a recompute) and lists the valid (next activity, next resource) pairs
    allowed after the case prefix;
  * scores every pair with the predictive models on three objectives -- case
    outcome probability (maximize), predicted total time (minimize, plotted as
    1 - time) and prediction uncertainty (minimize, plotted as a normalised
    "confidence" = 1 - uncertainty);
  * computes the Pareto front twice, once with the exhaustive search and once
    with NSGA-II, and records the wall-clock time of each;
  * highlights the top-k actions chosen by p-dispersion and the "no
    recommendation" baseline point (the case left as it happened in the log);
  * saves a single figure under ``save_dir`` with one subplot per method
    (exhaustive on the left, NSGA-II on the right) -- either a 2D plot with
    confidence encoded as point color (``--view color``, default) or a 3D
    scatter with confidence on the third axis (``--view 3d``).

Example usage:
    python 3_plot_pareto.py --case_study "BAC" --k 5
    python 3_plot_pareto.py --case_study "BAC" --k 5 --view 3d
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

from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.get_features import load_case_study, get_case_study_features
from utils.setup_cache import get_transition_graph
from utils.recommendation_functions import (
    act_with_res_func,
    build_query_instances,
    next_possible_activities,
    _to_row_df,
    nsga2_pareto_search,
    exhaustive_pareto_search,
    _build_valid_pairs,
    _evaluate_candidates,
    predict_time_and_uncertainty,
    select_top_k_pareto_actions,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

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
        "BPI12": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }

def evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model):
    """Evaluate every candidate (activity, resource) pair on the three objectives.

    Uses ``predict_proba`` for the outcome model when available so that the
    outcome objective is a probability rather than a hard 0/1 label.

    Input:
        valid_pairs: iterable of (next_activity, next_resource) tuples to score.
        query_instance: the prefix/query instance (row-like) describing the case
            state before applying a recommendation.
        predictive_outcome_model: fitted classifier for the case outcome; its
            ``predict_proba`` (preferred) or ``predict`` is called.
        predictive_time_model: fitted model returning the predicted total time
            and its uncertainty via ``predict_time_and_uncertainty``.
    Output:
        np.ndarray of shape (len(valid_pairs), 3) with columns
        [predicted_outcome, predicted_total_time, predicted_uncertainty] -- the
        same three objectives used by the Pareto search.
    """
    base_outcome_row = _to_row_df(query_instance).iloc[0].to_dict()
    base_time_row = _to_row_df(query_instance).iloc[0].to_dict()

    outcome_rows, time_rows = [], []
    for next_act, next_res in valid_pairs:
        o_row = dict(base_outcome_row)
        o_row['NEXT_ACTIVITY'] = next_act
        o_row['NEXT_RESOURCE'] = next_res
        outcome_rows.append(o_row)

        t_row = dict(base_time_row)
        t_row['NEXT_ACTIVITY'] = next_act
        t_row['NEXT_RESOURCE'] = next_res
        time_rows.append(t_row)

    df_out = pd.DataFrame(outcome_rows)
    if hasattr(predictive_outcome_model, "predict_proba"):
        predicted_outcome = predictive_outcome_model.predict_proba(df_out)[:, 1]
    else:
        predicted_outcome = predictive_outcome_model.predict(df_out)

    predicted_total_time, predicted_uncertainty = predict_time_and_uncertainty(
        predictive_time_model, pd.DataFrame(time_rows)
    )
    return np.column_stack([predicted_outcome, predicted_total_time, predicted_uncertainty])


def run_and_plot_comparison(
    case_study: str,
    target_case_id: str = None,
    window_size: int = 5,
    pop_size: int = 50,
    n_generations: int = 10,
    random_state: int = 1234,
    k: int = 5,
    view: str = "color",
    elev: float = 22.0,
    azim: float = 0,
    rebuild_cache: bool = False,
    save_dir: str = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\pareto_front_images"
):
    """Run the exhaustive and NSGA-II Pareto searches for one case and plot both.

    For the chosen case the function evaluates every valid (activity, resource)
    pair, computes the Pareto front with each method, highlights the top-k
    actions selected by p-dispersion, adds the "no recommendation" baseline
    point, and saves a single figure to ``save_dir`` with one subplot per
    method (exhaustive and NSGA-II side by side).

    Input:
        case_study: dataset name (e.g. "BAC", "BPI12", "bpi17_before").
        target_case_id: case id to analyse; if None or not present in the test
            set, the case with the largest / most spread-out Pareto front is
            selected automatically.
        window_size: prefix window length used to build the transition system
            and to look up the next possible activities.
        pop_size: NSGA-II population size.
        n_generations: number of NSGA-II generations.
        random_state: seed for numpy / random and for NSGA-II reproducibility.
        k: number of Pareto points to highlight as the top-k selection.
        view: "color" -> 2D plot (outcome vs 1-time) with confidence
            (1 - normalised uncertainty) mapped to point color; "3d" -> 3D
            scatter with confidence as the third axis.
        elev, azim: elevation and azimuth (degrees) of the 3D camera. The
            defaults keep the ideal point (1, 1, 1) at the top corner facing the
            viewer while making the outcome and 1-time axes easy to read;
            ignored when view != "3d".
        save_dir: directory where the output .jpg figure is written (created
            if missing).
    Output:
        None. The combined figure is written to disk and progress is printed
        to stdout.
        The function returns early (printing a message) if the case has no
        possible next activity or no valid action-resource pair.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    print(f"Loading data for {case_study}...")
    train_data, test_data, test_log = load_case_study(case_study)

    if case_study in {"BPI12"}:
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
    unique_cases = pd.unique(test_data[case_id_name])

    # Automatic selection of the case with the widest, most spread-out front
    if target_case_id is None or target_case_id not in unique_cases:
        print("Searching for the case with the most solutions and the most spread-out Pareto curve...")
        best_score = -1
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

            all_evals = evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
            all_x = all_evals[:, 0]
            all_y = 1.0 - all_evals[:, 1]

            pareto_vals = np.column_stack((all_x, all_y))
            try:
                is_pareto = paretoset(pareto_vals, sense=["max", "max"])
                front_x = all_x[is_pareto]
                front_y = all_y[is_pareto]

                spread_x = np.max(front_x) - np.min(front_x)
                spread_y = np.max(front_y) - np.min(front_y)
                spread_area = spread_x * spread_y
                score = n_solutions * spread_area

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

    # "No recommendation" point: evaluate the models on the query_instance as it
    # is, i.e. with NEXT_ACTIVITY/NEXT_RESOURCE equal to what actually happened
    # in the log (no recommended action applied).
    baseline_row = _to_row_df(query_instance)
    if hasattr(predictive_outcome_model, "predict_proba"):
        baseline_outcome = predictive_outcome_model.predict_proba(baseline_row)[:, 1][0]
    else:
        baseline_outcome = predictive_outcome_model.predict(baseline_row)[0]
    baseline_time_arr, baseline_unc_arr = predict_time_and_uncertainty(predictive_time_model, baseline_row)
    baseline_time = float(baseline_time_arr[0])
    baseline_unc = float(baseline_unc_arr[0])
    baseline_x = baseline_outcome
    baseline_y = 1.0 - baseline_time

    # =========================================================================
    methods = ["exhaustive", "nsga2"]

    # Single figure with one subplot per method (side by side), saved as one file.
    os.makedirs(save_dir, exist_ok=True)
    if view == "3d":
        # A near-square area per subplot wastes far less space around a 3D cube
        # than a wide one; wspace still needs to be large enough that the right
        # cube's (inner) z-axis and its tick labels are not hidden behind the
        # left cube.
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw={"projection": "3d"},
                                 gridspec_kw={"wspace": 0.22})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, method in zip(axes, methods):
        print(f"\nRunning method: {method.upper()}...")
        t_start = time.time()

        all_evals = evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
        if method == "exhaustive":
            pareto_set = exhaustive_pareto_search(query_instance, poss, predictive_outcome_model, predictive_time_model, act_with_res)
        else:  # nsga2
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

        all_x = all_evals[:, 0]
        all_y = 1.0 - all_evals[:, 1]
        all_unc = all_evals[:, 2]

        if not pareto_set:
            print(f"Empty Pareto set for {method}. Skipping this subplot.")
            ax.set_title(f"{method.upper()}\n(empty Pareto set)")
            ax.set_axis_off()
            continue

        # Confidence = 1 - uncertainty, min-max normalised over all evaluated
        # points (baseline included), so 0 = least reliable point, 1 = most
        # reliable. Used only for the point color / the third plot axis.
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

        title_text = (
            f"{method.upper()}\n"
            f"Execution Time: {elapsed_time:.4f} seconds"
        )

        if view == "3d":
            ax.scatter(all_x, all_y, all_conf, color="black", alpha=0.35, s=30, label="Evaluated Pairs (All)")
            ax.scatter(front_x, front_y, front_conf, color="blue", s=80, label="Pareto Front")
            ax.scatter(top_k_x, top_k_y, top_k_conf, color="purple", marker="P", s=180,
                       edgecolors="black", linewidths=0.6, label=f"Top-{k} Selected (p-dispersion)")
            ax.scatter([1.0], [1.0], [1.0], color="green", marker="X", s=180, label="Ideal Point (1,1,1)")
            ax.scatter([baseline_x], [baseline_y], [baseline_conf], color="orange", marker="D", s=150,
                       label="No Recommendation (Baseline)")
            ax.set_xlabel("Predicted Outcome (max)")
            ax.set_ylabel("1 - Predicted Time (max)")
            ax.set_zlabel("Confidence = 1 - norm(uncertainty) (max)")
            # Axis limits are left to matplotlib's autoscaling so the plot zooms
            # onto the region the points actually occupy (the ideal-point marker
            # keeps (1, 1, 1) inside the view).
            # Draw the "1 - Predicted Time" axis with 0 on the right (next to the
            # confidence axis) and 1 on the left, so it grows away from the
            # "Predicted Outcome" axis instead of sharing its far corner.
            
            # Orientation chosen so the ideal point (1, 1, 1) sits at the top
            # corner toward the viewer and the outcome / 1-time axes stay
            # readable; tune with --elev / --azim.
            ax.view_init(elev=elev, azim=azim)
            # Enlarge the drawn cube inside its axes rectangle -- by default
            # matplotlib leaves a wide empty margin around a 3D plot.
            ax.set_box_aspect(None, zoom=1.25)
            ax.set_title(title_text)
        else:  # "color": 2D plot, confidence encoded as point color
            ax.scatter(all_x, all_y, c=all_conf, cmap="viridis", vmin=0.0, vmax=1.0,
                       alpha=0.55, s=35)
            sc = ax.scatter(front_x, front_y, c=front_conf, cmap="viridis", vmin=0.0, vmax=1.0,
                            s=95, edgecolors="black", linewidths=1.4, zorder=5)
            ax.plot(front_x, front_y, color="grey", linestyle="--", alpha=0.5, zorder=4)
            ax.scatter(top_k_x, top_k_y, facecolors="none", edgecolors="crimson", marker="P",
                       s=190, linewidths=1.9, zorder=8)
            ax.scatter(1.0, 1.0, color="green", marker="X", s=110, zorder=10)
            ax.scatter(baseline_x, baseline_y, color="orange", marker="D", s=110,
                       edgecolors="black", linewidths=0.6, zorder=10)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Confidence = 1 - norm(uncertainty)  (higher is better)")

            ax.set_xlabel("Predicted Outcome (Probability Maximize)")
            ax.set_ylabel("1 - Predicted Time (Maximize)")
            ax.set_title(title_text)
            ax.grid(True, linestyle=":", alpha=0.7)

            plot_x_min = min(np.min(all_x), baseline_x)
            plot_x_max = max(np.max(all_x), baseline_x)
            plot_y_min = min(np.min(all_y), baseline_y)
            plot_y_max = max(np.max(all_y), baseline_y)
            margin_x = (plot_x_max - plot_x_min) * 0.05 if plot_x_max != plot_x_min else 0.05
            margin_y = (plot_y_max - plot_y_min) * 0.05 if plot_y_max != plot_y_min else 0.05
            ax.set_xlim(min(plot_x_min - margin_x, -0.05), max(plot_x_max + margin_x, 1.05))
            ax.set_ylim(min(plot_y_min - margin_y, -0.05), max(plot_y_max + margin_y, 1.05))

        print(f"  ({method.upper()} done in {elapsed_time:.4f}s)")

    # Both methods drawn -> save the combined figure as a single file.
    fig.suptitle(
        f"Pareto Front Analysis  |  Dataset: {case_study}  |  Case ID: {target_case_id}",
        fontsize=13,
    )

    # One shared legend for the whole figure, laid out horizontally under both
    # subplots -- avoids repeating the same box twice and frees up plot area.
    ideal_label = "Ideal point (1, 1, 1)" if view == "3d" else "Ideal point (1, 1)"
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="grey",
                      markersize=8, label="Evaluated pairs (color = confidence)"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="grey",
                      markeredgecolor="black", markersize=9, label="Pareto front"),
        mlines.Line2D([], [], marker="P", color="none", markeredgecolor="crimson",
                      markerfacecolor="none", markersize=13, label=f"Top-{k} selected (p-dispersion)"),
        mlines.Line2D([], [], marker="X", color="none", markerfacecolor="green",
                      markersize=11, label=ideal_label),
        mlines.Line2D([], [], marker="D", color="none", markerfacecolor="orange",
                      markeredgecolor="black", markersize=9, label="No recommendation (baseline)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    # tight_layout misbehaves with 3D axes (it collapses the wspace we set), so
    # only use it for the 2D view; reserve a strip at the bottom for the legend.
    if view != "3d":
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    else:
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.10, top=0.90, wspace=0.22)
    filename = f"pareto_{case_study}_{str(target_case_id).replace(':', '_')}_{view}.jpg"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Pareto comparison with execution times.')
    parser.add_argument('--case_study', type=str, required=True, help='Dataset name')
    parser.add_argument('--case_id', type=str, default=None, help='Specific case ID (optional)')
    parser.add_argument('--k', type=int, default=5, help='Number of top-k points to highlight (default: 5)')
    parser.add_argument('--view', type=str, default='color', choices=['color', '3d'],
                        help="'color' = 2D plot with confidence as point color (default); '3d' = 3D scatter")
    parser.add_argument('--elev', type=float, default=22.0,
                        help="3D camera elevation in degrees (default: 22; only used with --view 3d)")
    parser.add_argument('--azim', type=float, default=-45.0,
                        help="3D camera azimuth in degrees (default: -90; only used with --view 3d)")
    parser.add_argument('--rebuild-cache', dest='rebuild_cache', action='store_true',
                        help="Force recomputing the (cached) transition system instead of loading it")

    try:
        args = parser.parse_args()
        run_and_plot_comparison(case_study=args.case_study, target_case_id=args.case_id, k=args.k,
                                view=args.view, elev=args.elev, azim=args.azim,
                                rebuild_cache=args.rebuild_cache)
    except Exception as e:
        print(f"Error during execution: {e}")


# example usage:
# python 3_plot_pareto.py --case_study "BAC" --k 5
# python 3_plot_pareto.py --case_study "BAC" --k 5 --view 3d
# python 3_plot_pareto.py --case_study "BAC" --k 5 --view 3d --elev 30 --azim 45
