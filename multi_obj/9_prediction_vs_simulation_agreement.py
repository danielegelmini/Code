#!/usr/bin/env python3
"""
How often what the predictive models promise for a recommended action (step 3) also happens in the
simulation (step 4), for every recommended point (case, rank) of the evaluation tables built by
5_result_computation.py: case_studies/<cs>/evaluation_tables/<method>_all_ranks.csv (one row per point,
predictions and simulated means) and <method>_runs.csv (the single simulation runs, rank 0 = baseline).

Every comparison is against the "no recommendation" baseline, on its own side: the prediction with the
recommendation against the prediction without it, the simulated runs with the recommendation against
the simulated baseline runs.

Changes of the two KPIs:
  - outcome: absolute change of the probability of a positive outcome, in percentage points
    (predicted probability; share of positive runs in the simulation);
  - remaining time: relative change, in % of the baseline remaining time (negative = shorter).
"Improves" means a strictly better value (outcome higher, time shorter), except that in the simulation
an outcome with all runs positive both with and without the recommendation counts as improved: it
cannot do better.

Reported, for each case study, first as detailed tables (sections 1-3, on all the recommended points),
then as a compact SUMMARY:
  1. per KPI: of the points whose prediction improves the KPI, how many improve it in the simulation too
     (2x2 table predicted improves yes/no x simulated improves yes/no), against the reference of the points
     whose prediction does NOT improve it but the simulation does anyway: if the model's yes/no carried no
     information the two shares would be equal, so their difference (yes - no) measures how informative it is;
  2. the same for the points improving BOTH KPIs;
  3. the same for the "good" points: a point is good when it improves one KPI by at least --x (pp for the
     outcome, % for the time) and worsens the other by at most --y; judged with the same rule on the
     predicted and on the simulated changes. (x = 0, y = 0 gives back "improves both"; x = 0 with no limit
     on y "improves at least one".)
  4. expected value with the recommendation, E_p { E_r [ (sum_i S_i) / n_sim ] } (averaged over the ranks of
     a prefix, then over the prefixes), on all the points and on the good points only (as if only those were
     recommended, with the share of prefixes left with at least one) -- four values, simulated and predicted
     for each KPI, where (sum_i S_i) / n_sim is:
       outcome, simulated: the share of positive runs (S_i = 1 if run i is positive);
       outcome, predicted: the predicted probability of a positive outcome;
       time, simulated:    the mean remaining time over the runs (S_i = remaining time of run i);
       time, predicted:    the predicted remaining time;
     all with the recommendation, times in days.
(Step 3's rank numbers are only the order of the candidate list, not a preference: no rank is analysed apart.)

Writes 9_prediction_vs_simulation_agreement.txt (the report) and .csv (the numbers, one row per case study,
selection and quantity) in this folder, and the per-point values and flags in
case_studies/<cs>/evaluation_tables/<method>_agreement_points.csv (which also has sim_time_win_share, the
share of the (run with the recommendation, baseline run) pairs won by the recommendation, not in the report).

Usage:
    python 9_prediction_vs_simulation_agreement.py
    python 9_prediction_vs_simulation_agreement.py --case_studies "BPI12,BPI12_sim" --x 5 --y 10
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils.simulation_functions import case_id_name
from utils.predictive_models_functions import SECONDS_PER_DAY

HERE = Path(__file__).resolve().parent
DEFAULT_CASE_STUDIES = "BAC,BPI12,bpi17_before,bpi17_after"
EVAL_TABLES_SUBDIR = "evaluation_tables"
OUT_STEM = "9_prediction_vs_simulation_agreement"


def load_tables(base_dir: Path, case_study: str, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two evaluation tables of one case study written by 5_result_computation.py.

    Args:
        base_dir: directory containing case_studies/.
        case_study: case study name, e.g. "BAC".
        method: recommendation method, the prefix of the file names, e.g. "exhaustive".

    Returns:
        points: <method>_all_ranks.csv, one row per recommended point (case, rank): predictions with and
            without the recommendation, simulated means with and without it.
        runs: <method>_runs.csv, one row per single simulation run (rank, case, run, status, remaining_time,
            applied); rank 0 = the baseline runs, without recommendation.

    Raises:
        FileNotFoundError: one of the two files is missing.
    """
    table_dir = base_dir / "case_studies" / case_study / EVAL_TABLES_SUBDIR
    points_path, runs_path = table_dir / f"{method}_all_ranks.csv", table_dir / f"{method}_runs.csv"
    for path in (points_path, runs_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run 5_result_computation.py --case_studies {case_study}")
    # case ids read as text in both tables, so that they match when the runs are looked up by case
    points = pd.read_csv(points_path, dtype={case_id_name: str})
    runs = pd.read_csv(runs_path, dtype={case_id_name: str})
    return points, runs


def relative_change_pct(with_rec: pd.Series, without: pd.Series) -> pd.Series:
    """100 * (with - without) / without; 0 when both are 0, +inf when only the baseline is 0.

    Args:
        with_rec: remaining time with the recommendation, one value per point.
        without: remaining time without it (the baseline), same index.

    Returns:
        The relative change in % of the baseline, same index (negative = shorter with the recommendation).
    """
    with_rec, without = with_rec.astype(float), without.astype(float)
    # a zero baseline becomes NaN in the denominator (no division by zero), then filled in below
    out = 100.0 * (with_rec - without) / without.where(without != 0)
    out[(without == 0) & (with_rec == 0)] = 0.0
    out[(without == 0) & (with_rec > 0)] = np.inf
    return out


def kpi_flags(d_out: pd.Series, d_time: pd.Series, ceiling: pd.Series, x: float, y: float) -> pd.DataFrame:
    """Improvement flags from the changes (outcome in pp, time in %), ceiling = outcome already 1 both ways.

    Args:
        d_out: change of the outcome, in percentage points (with - without recommendation), one value per point.
        d_time: relative change of the remaining time, in % of the baseline (negative = shorter).
        ceiling: True where the outcome is already 1 both with and without the recommendation (counts as
            improved); all False on the predicted side.
        x: minimum improvement of one KPI for a good point (pp for the outcome, % for the time).
        y: maximum worsening of the other KPI for a good point.

    Returns:
        One boolean row per point, columns: outcome (outcome improves), time (time improves),
        both (both improve), good (good point).
    """
    # strict improvements: outcome higher, time shorter
    out_improves = (d_out > 0) | ceiling
    time_improves = d_time < 0
    # improvements by at least x
    out_by_x = ((d_out > 0) & (d_out >= x)) | ceiling
    time_by_x = (d_time < 0) & (d_time <= -x)
    # good: outcome up by >= x with the time longer by <= y %, or time shorter by >= x % with the outcome down by <= y pp
    good = (out_by_x & (d_time <= y)) | (time_by_x & (d_out >= -y))
    return pd.DataFrame({"outcome": out_improves, "time": time_improves,
                         "both": out_improves & time_improves, "good": good})


def per_point_simulation(points: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per point: the share of the (run with the recommendation, baseline run) pairs in which the run with
    the recommendation has the shorter remaining time (ties 1/2).

    Args:
        points: one row per recommended point, with the columns rank and case id.
        runs: one row per simulation run (rank 0 = baseline), with applied and remaining_time.

    Returns:
        One column, sim_time_win_share, same index as points; NaN where the runs of the point or of its
        baseline are missing.
    """
    runs = runs[runs["applied"].astype(bool)]  # only the runs in which the action was actually applied
    # (rank, case id) -> the runs of that point; (0, case id) -> the baseline runs of that case
    by_key = {key: g for key, g in runs.groupby(["rank", case_id_name])}
    rows = []
    for rank, cid in zip(points["rank"], points[case_id_name]):
        rec, base = by_key.get((rank, cid)), by_key.get((0, cid))  # runs with the recommendation, baseline runs
        if rec is None or base is None or rec.empty or base.empty:
            rows.append({"sim_time_win_share": np.nan})
            continue
        t_rec, t_base = rec["remaining_time"].to_numpy(float), base["remaining_time"].to_numpy(float)
        diff = t_rec[:, None] - t_base[None, :]  # every (run with recommendation, baseline run) pair
        # win = 1, tie = 1/2, loss = 0, averaged over all the pairs
        win_share = float(np.mean((diff < 0) + 0.5 * (diff == 0)))
        rows.append({"sim_time_win_share": win_share})
    return pd.DataFrame(rows, index=points.index)


def build_points(points: pd.DataFrame, runs: pd.DataFrame, x: float, y: float) -> pd.DataFrame:
    """The per-point changes, flags and simulation statistics.

    Args:
        points: the points table from load_tables, one row per recommended point.
        runs: the runs table from load_tables, one row per simulation run.
        x, y: the thresholds of a good point (see kpi_flags).

    Returns:
        The points with simulated means, plus: the four changes (pred_/sim_d_outcome_pp, pred_/sim_d_time_pct),
        the flags pred_/sim_improves_{outcome,time,both}, pred_good, sim_good, and sim_time_win_share.
    """
    # only the points that were simulated (with and without the recommendation)
    p = points.dropna(subset=["sim_status_method_mean", "sim_remaining_time_method_mean",
                              "sim_status_baseline_mean", "sim_remaining_time_baseline_mean"]).copy()
    # changes against the baseline, each on its own side: predicted with vs without, simulated with vs without
    # compute the difference between predicted/simulated with/without in percentage 
    p["pred_d_outcome_pp"] = 100.0 * (p["pred_status_with_rec"] - p["pred_status_no_rec"])
    p["pred_d_time_pct"] = relative_change_pct(p["pred_remaining_time_with_rec"], p["pred_remaining_time_no_rec"])
    p["sim_d_outcome_pp"] = 100.0 * (p["sim_status_method_mean"] - p["sim_status_baseline_mean"])
    p["sim_d_time_pct"] = relative_change_pct(p["sim_remaining_time_method_mean"], p["sim_remaining_time_baseline_mean"])

    # the ceiling (all runs positive both ways) exists only in the simulation: a predicted probability is never 1
    no_ceiling = pd.Series(False, index=p.index)
    sim_ceiling = (p["sim_status_method_mean"] == 1) & (p["sim_status_baseline_mean"] == 1)
    # compute the point that are good for one kpi, both and the "good" ones
    pred_flags = kpi_flags(p["pred_d_outcome_pp"], p["pred_d_time_pct"], no_ceiling, x, y).add_prefix("pred_improves_")
    sim_flags = kpi_flags(p["sim_d_outcome_pp"], p["sim_d_time_pct"], sim_ceiling, x, y).add_prefix("sim_improves_")
    p = pd.concat([p, pred_flags, sim_flags, per_point_simulation(p, runs)], axis=1)
    return p.rename(columns={"pred_improves_good": "pred_good", "sim_improves_good": "sim_good"})


def agreement(pred: pd.Series, sim: pd.Series) -> dict:
    """2x2 counts of predicted yes/no x simulated yes/no and the shares built on them.

    Args:
        pred: one boolean per point, the flag on the predicted side (e.g. pred_improves_outcome).
        sim: the same flag on the simulated side, same index.

    Returns:
        A dict: n_points; the four cells of the 2x2 table (pred_yes_sim_yes, ...); pred_yes, pred_no and
        pred_yes_pct; sim_yes_given_pred_yes_pct (confirmed by the simulation), sim_yes_given_pred_no_pct
        (the reference) and their difference yes_minus_no_pp; pred_yes_sim_yes_of_all_pct.
    """
    n = len(pred)
    # first letter = predicted, second = simulated (y = yes, n = no); ~ = not
    yy, yn = int((pred & sim).sum()), int((pred & ~sim).sum())
    ny, nn = int((~pred & sim).sum()), int((~pred & ~sim).sum())
    n_pred, n_pred_no = yy + yn, ny + nn  # points predicted yes, predicted no
    # NaN instead of a division by zero when a row of the table is empty
    yes_pct = 100.0 * yy / n_pred if n_pred else np.nan
    no_pct = 100.0 * ny / n_pred_no if n_pred_no else np.nan
    return {"n_points": n, 
            "pred_yes": n_pred, 
            "pred_yes_pct": 100.0 * n_pred / n if n else np.nan,
            "pred_yes_sim_yes": yy, 
            "pred_yes_sim_no": yn, 
            "pred_no_sim_yes": ny, 
            "pred_no_sim_no": nn,
            "sim_yes_given_pred_yes_pct": yes_pct,
            "pred_yes_sim_yes_of_all_pct": 100.0 * yy / n if n else np.nan,
            "pred_no": n_pred_no, 
            "sim_yes_given_pred_no_pct": no_pct,
            "yes_minus_no_pp": yes_pct - no_pct,
            }


def expected_over_prefixes(p: pd.DataFrame, column: str) -> float:
    """E_p { E_r [column] }: mean over the ranks of each prefix, then over the prefixes.

    Args:
        p: the points (one row per (case, rank)).
        column: the per-point value to average.

    Returns:
        The mean, a float; every prefix weighs the same however many ranks it has.
    """
    # groupby(...).mean(): one value per prefix (mean over its ranks); the last .mean(): over the prefixes
    return float(p.groupby(case_id_name)[column].mean().mean())


def format_agreement(title: str, a: dict, what: str = "improves") -> list[str]:
    """Report lines of one 2x2 table.

    Args:
        title: heading of the table, e.g. "remaining time".
        a: the dict returned by agreement.
        what: the word used in the lines ("improves" or "good").

    Returns:
        The lines: counts and shares, the reference, the 2x2 table, an empty line.
    """
    lines = [f"  {title}",
             f"    predicted {what} : {a['pred_yes']:5d} of {a['n_points']} points ({a['pred_yes_pct']:.1f}%)",
             f"    of these, simulated {what} too: {a['pred_yes_sim_yes']:5d} ({a['sim_yes_given_pred_yes_pct']:.1f}%)"
             f"  -- {a['pred_yes_sim_yes_of_all_pct']:.1f}% of all points"]
    lines.append(f"    reference -- predicted NOT {what} ({a['pred_no']} points), simulated {what} anyway: "
                 f"{a['pred_no_sim_yes']:5d} ({a['sim_yes_given_pred_no_pct']:.1f}%)  "
                 f"=> yes - no = {a['yes_minus_no_pp']:+.1f} pp (0 = the model's yes/no says nothing)")
    lines += ["                              simulated yes   simulated no",
              f"    predicted yes          {a['pred_yes_sim_yes']:15d} {a['pred_yes_sim_no']:14d}",
              f"    predicted no           {a['pred_no_sim_yes']:15d} {a['pred_no_sim_no']:14d}", ""]
    return lines


def detailed_tables(p: pd.DataFrame, x: float, y: float):
    """Sections 1-3 on all the points: the 2x2 tables, with the reference shares.

    Args:
        p: the points from build_points (changes and flags).
        x, y: the thresholds of a good point, only written in the section 3 title.

    Returns:
        lines: the report lines of sections 1-3.
        rows: one CSV row (dict) per table, with the agreement values.
        summary: the agreement dict of each table, keyed "outcome", "time", "both", "good" (for the SUMMARY).
    """
    lines, rows = [], []
    summary = {}

    def add(section, a):  # one CSV row for the table of a section
        rows.append({"selection": "all", "section": section, **a})

    lines.append("1. Per KPI")
    for kpi, label in (("outcome", "outcome (probability of a positive outcome)"), ("time", "remaining time")):
        a = agreement(p[f"pred_improves_{kpi}"], p[f"sim_improves_{kpi}"])
        lines += format_agreement(label, a)
        add(f"1_{kpi}", a)
        summary[kpi] = a

    lines.append("2. Both KPIs")
    a = agreement(p["pred_improves_both"], p["sim_improves_both"])
    lines += format_agreement("improves outcome AND remaining time", a)
    add("2_both", a)
    summary["both"] = a

    lines.append(f"3. Good points (one KPI improved by >= {x:g} -- pp / % -- the other worsened by <= {y:g})")
    a = agreement(p["pred_good"], p["sim_good"])
    lines += format_agreement("good", a, what="good")
    add("3_good", a)
    summary["good"] = a
    return lines, rows, summary


def expected_values(p: pd.DataFrame) -> dict:
    """Section 4, E_p{ E_r[...] } on the points p (mean over the ranks of a prefix, then over the prefixes),
    with the recommendation, simulated and predicted --
    outcome: share of positive runs (predicted: probability of a positive outcome);
    time: mean remaining time over the runs (predicted: the predicted remaining time), in days.

    Args:
        p: the points to average over (all of them, or the good ones only).

    Returns:
        A dict: out_sim, out_pred (between 0 and 1), time_sim_days, time_pred_days.
    """
    e = lambda c: expected_over_prefixes(p, c)  # shorthand: E_p{E_r[column c]}
    # each quantity simulated and predicted, side by side
    return {"out_sim": e("sim_status_method_mean"),
            "out_pred": e("pred_status_with_rec"),
            "time_sim_days": e("sim_remaining_time_method_mean") / SECONDS_PER_DAY,
            "time_pred_days": e("pred_remaining_time_with_rec") / SECONDS_PER_DAY,
            }


def formula_lines(expected: dict, coverage: dict, n_prefixes: int) -> list[str]:
    """Section 4 in detail: the four expected values, simulated vs predicted.

    Args:
        expected: {"all": ..., "good": ...}, the dicts returned by expected_values.
        coverage: {"all": ..., "good": ...}, number of prefixes with at least one point of the selection.
        n_prefixes: number of prefixes of the case study.

    Returns:
        The report lines of section 4, ending with an empty line.
    """
    lines = ["4. E_p{ E_r[ (sum_i S_i) / n_sim ] } with the recommendation -- simulation vs prediction"]
    for selection, title in (("all", "all points"), ("good", "good points only")):
        v = expected[selection]
        lines.append(f"  {title} ({coverage[selection]} of {n_prefixes} prefixes)")
        for label, sim_key, pred_key in (
                ("outcome: share of positive runs / predicted probability", "out_sim", "out_pred"),
                ("remaining time, days: mean over the runs / predicted", "time_sim_days", "time_pred_days")):
            lines.append(f"    {label:58s} simulated {v[sim_key]:8.3f}   predicted {v[pred_key]:8.3f}   "
                         f"difference {v[sim_key] - v[pred_key]:+.3f}")
    return lines + [""]


def summary_block(summary: dict, expected: dict, coverage: dict, n_prefixes: int, x: float, y: float) -> list[str]:
    """The compact summary of a case study.

    Args:
        summary: the agreement dicts from detailed_tables ("outcome", "time", "both", "good").
        expected, coverage, n_prefixes: as in formula_lines.
        x, y: the thresholds of a good point, only written in the text.

    Returns:
        The SUMMARY lines, ending with an empty line.
    """
    lines = ["SUMMARY",
             "1-2. Is the predicted improvement seen in the simulation too? (all points)",
             "                 predicted        confirmed   if predicted NO,      discrimination",
             "                 improves         by sim      sim improves anyway   (yes - no)"]
    for kpi in ("outcome", "time", "both"):
        a = summary[kpi]
        # fields padded to start under the header's columns (17, 34, 46, 68)
        lines.append(f"   {kpi:13s} " + f"{a['pred_yes']:5d} ({a['pred_yes_pct']:3.0f}%)".ljust(17)
                     + f"{a['sim_yes_given_pred_yes_pct']:3.0f}%".ljust(12)
                     + f"{a['sim_yes_given_pred_no_pct']:3.0f}%".ljust(22)
                     + f"{a['yes_minus_no_pp']:+4.0f}")
    g = summary["good"]
    lines += ["",
              f"3. Good points (x = {x:g}, y = {y:g})",
              f"   {g['pred_yes']} good points ({g['pred_yes_pct']:.0f}%), in {coverage['good']} of {n_prefixes} prefixes "
              f"({100.0 * coverage['good'] / n_prefixes:.0f}%); good in the simulation too: {g['sim_yes_given_pred_yes_pct']:.0f}%",
              "",
              "4. Expected value with the recommendation (formula E_p{E_r[...]})",
              "   outcome: share of positive runs / predicted probability; time: remaining time in days",
              "                                  all points   good points only"]
    for kpi, sim_key, pred_key, fmt in (("outcome", "out_sim", "out_pred", ".3f"),
                                        ("time, days", "time_sim_days", "time_pred_days", ".2f")):
        for side, key in (("simulated", sim_key), ("predicted", pred_key)):
            name = kpi if side == "simulated" else ""
            lines.append(f"   {name:12s} {side:12s} {expected['all'][key]:12{fmt}} {expected['good'][key]:14{fmt}}")
    # share of the prefixes with at least one point of the selection
    lines.append(f"   {'prefixes covered':25s} {100.0 * coverage['all'] / n_prefixes:11.0f}% "
                 f"{100.0 * coverage['good'] / n_prefixes:13.0f}%")
    return lines + [""]


def analyse_case_study(case_study: str, p: pd.DataFrame, x: float, y: float):
    """Report lines (detailed tables, then the summary) and CSV rows of one case study.

    Args:
        case_study: case study name, written in the report and in the CSV rows.
        p: the points from build_points.
        x, y: the thresholds of a good point.

    Returns:
        lines: the report lines of the case study (sections 1-4, then the SUMMARY).
        rows: the CSV rows (dicts), each with the case_study field.
    """
    n_prefixes = p[case_id_name].nunique()  # distinct cases = prefixes
    lines = [f"{case_study}  ({len(p)} recommended points, {n_prefixes} prefixes)", ""]
    detail_lines, rows, summary = detailed_tables(p, x, y)
    # the points judged good by the prediction: the ones that would be recommended under the good rule
    good = p[p["pred_good"]]
    coverage = {"all": n_prefixes, "good": good[case_id_name].nunique()}
    # no good point: all the section 4 values NaN, with the same keys
    expected = {"all": expected_values(p), "good": expected_values(good) if not good.empty else {
        k: np.nan for k in expected_values(p)}}
    for selection, values in expected.items():  # one CSV row per selection for section 4
        rows.append({"selection": selection, "section": "4_expected_values", "n_points": len(p if selection == "all" else good),
                     "prefixes": coverage[selection], **values})
    lines += (detail_lines + formula_lines(expected, coverage, n_prefixes)
              + summary_block(summary, expected, coverage, n_prefixes, x, y))
    return lines, [{"case_study": case_study, **r} for r in rows]


def main():
    """Parse the command line, analyse every case study, write the report (.txt), the numbers (.csv) and the
    per-point tables."""
    parser = argparse.ArgumentParser(description="Agreement between the predicted and the simulated effect of "
                                                 "the recommended actions (see the module docstring).")
    parser.add_argument("--base_dir", type=str, default=".", help="Directory containing case_studies/ (default: .)")
    parser.add_argument("--case_studies", type=str, default=DEFAULT_CASE_STUDIES,
                        help=f'Comma-separated case studies (default: "{DEFAULT_CASE_STUDIES}")')
    parser.add_argument("--method", type=str, default="exhaustive")
    parser.add_argument("--x", type=float, default=5.0,
                        help="Good point: minimum improvement of one KPI (pp for the outcome, %% for the time)")
    parser.add_argument("--y", type=float, default=5.0,
                        help="Good point: maximum worsening of the other KPI (pp for the outcome, %% for the time)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    # the report, one string per line, joined and written at the end
    lines = ["=" * 110,
             "PREDICTED vs SIMULATED EFFECT OF THE RECOMMENDED ACTIONS",
             f"Written by 9_prediction_vs_simulation_agreement.py - {datetime.now():%Y-%m-%d %H:%M:%S} | "
             f"method {args.method} | good points x = {args.x:g}, y = {args.y:g}",
             "=" * 110,
             "Changes against the no-recommendation baseline, each on its own side. Outcome: percentage points of the",
             "probability of a positive outcome; time: % of the baseline remaining time. Simulation: means over the runs;",
             "an outcome positive in every run both with and without the recommendation counts as improved.",
             ""]
    rows = []  # the rows of the .csv, one dict per (case study, selection, section)
    for cs in [c.strip() for c in args.case_studies.split(",") if c.strip()]:
        try:
            points, runs = load_tables(base_dir, cs, args.method)
        except (FileNotFoundError, ValueError) as e:  # missing tables: skip this case study, go on with the others
            print(f"[SKIPPED] {cs}: {e}")
            lines += [f"{cs}: skipped ({e})", ""]
            continue
        p = build_points(points, runs, args.x, args.y)
        cs_lines, cs_rows = analyse_case_study(cs, p, args.x, args.y)
        lines += cs_lines + ["-" * 110, ""]
        rows += cs_rows
        # per-point values and flags, next to the evaluation tables
        points_path = base_dir / "case_studies" / cs / EVAL_TABLES_SUBDIR / f"{args.method}_agreement_points.csv"
        p.to_csv(points_path, index=False)
        print(f"[OK] {cs}: {len(p)} points -> {points_path}")

    (HERE / f"{OUT_STEM}.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(HERE / f"{OUT_STEM}.csv", index=False)
    print("\n".join(lines))
    print(f"Saved {HERE / (OUT_STEM + '.txt')} and {HERE / (OUT_STEM + '.csv')}")


if __name__ == "__main__":
    main()
