#!/usr/bin/env python3
"""
8d_aggregate_pred_vs_sim.py

Aggregate version of 8b_plot_case_comparison_vs_sim_model.py: instead of three hand-picked cases, ALL the test
cases and ranks of the evaluation table of `case_study` (built by 5_result_computation.py). For every
(case, rank) row and for the no-recommendation baseline of every case it compares

    real   : prediction of the models trained on the real training set (already in the table)
    sim    : prediction of the models trained on the fully simulated training set (--sim_case_study),
             computed here on the same query instance and the same recommended (activity, resource)
    ProSiT : simulated continuation, mean over the n_sim runs of step 4 (already in the table)

on the outcome (probability vs share of positive runs) and on the remaining time (days), both as levels and as
the effect of the recommendation (recommendation minus no-recommendation baseline, the quantity the recommender
acts on). It also reports the outcome noise of a 10-run mean (binomial), to tell model error from simulation noise,
and the training sets of the two case studies (cases, % positive, remaining time).

Writes, in this folder: 8d_pred_vs_sim_<case_study>.txt

Usage
-----
    python 8d_aggregate_pred_vs_sim.py
    python 8d_aggregate_pred_vs_sim.py --case_study BPI12 --sim_case_study BPI12_sim --n_sim 10
"""
import argparse
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils.get_features import load_case_study, get_case_study_features
from utils.pre_processing_functions import convert_dtypes_bpi12, NO_NEXT_TOKEN
from utils.predictive_models_functions import SECONDS_PER_DAY
from utils.recommendation_functions import build_query_instances

HERE = Path(__file__).resolve().parent
P8B = importlib.import_module("8b_plot_case_comparison_vs_sim_model")  # predict_batch, dtype handling


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def compare(name, sim, real, simm):
    """One line per model: MAE / bias / Pearson r against the simulated values."""
    out = []
    for label, pred in (("real model", real), ("BPI12_sim model", simm)):
        out.append({"quantity": name, "model": label, "n": len(sim),
                    "MAE vs ProSiT": float(np.mean(np.abs(pred - sim))),
                    "bias (pred - ProSiT)": float(np.mean(pred - sim)),
                    "r vs ProSiT": pearson(pred, sim)})
    out.append({"quantity": name, "model": "real vs BPI12_sim model", "n": len(sim),
                "MAE vs ProSiT": float(np.mean(np.abs(real - simm))),
                "bias (pred - ProSiT)": float(np.mean(real - simm)), "r vs ProSiT": pearson(real, simm)})
    return out


def training_stats(case_study):
    d = pd.read_csv(HERE / "case_studies" / case_study / "train_data.csv",
                    usecols=["case:concept:name", "label", "remaining_time", "total_time"])
    cases = d.groupby("case:concept:name").first()
    return {"case study": case_study, "cases": len(cases), "events": len(d),
            "% positive": 100 * cases["label"].mean(),
            "case duration median (d)": cases["total_time"].median() / SECONDS_PER_DAY,
            "case duration mean (d)": cases["total_time"].mean() / SECONDS_PER_DAY,
            "remaining time mean (d)": d["remaining_time"].mean() / SECONDS_PER_DAY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_study", default="BPI12")
    ap.add_argument("--sim_case_study", default="BPI12_sim")
    ap.add_argument("--method", default="exhaustive")
    ap.add_argument("--n_sim", type=int, default=10, help="runs per case of step 4 (for the noise estimate)")
    args = ap.parse_args()

    table = pd.read_csv(HERE / "case_studies" / args.case_study / "evaluation_tables" / f"{args.method}_all_ranks.csv")
    cid = "case:concept:name"
    table = table[np.isfinite(table["sim_status_method_mean"]) & np.isfinite(table["sim_remaining_time_method_mean"])
                  & np.isfinite(table["sim_status_baseline_mean"])].copy()
    table[cid] = table[cid].astype(str)

    _, test_data, _ = load_case_study(args.case_study)
    if args.case_study in P8B.BPI12_DTYPE_CASE_STUDIES:
        test_data = convert_dtypes_bpi12(test_data, "experiment")
    outcome_model, time_model, cid_name, *_ = get_case_study_features(args.sim_case_study)
    qi_by_case = {str(c): qi for c, qi in build_query_instances(test_data, cid_name).items()}
    table = table[table[cid].isin(qi_by_case)].copy()

    # recommendations and baselines through the BPI12_sim models, in one batch each
    res = [str(int(r)) if pd.notna(r) else r for r in table["rec_resource"]]
    st, rt = P8B.predict_batch([qi_by_case[c] for c in table[cid]], table["rec_activity"].tolist(), res,
                               outcome_model, time_model)
    table["simmodel_status_with_rec"], table["simmodel_rt_with_rec_d"] = st, rt
    base = table.groupby(cid).first().reset_index()
    bst, brt = P8B.predict_batch([qi_by_case[c] for c in base[cid]], [NO_NEXT_TOKEN] * len(base),
                                 [NO_NEXT_TOKEN] * len(base), outcome_model, time_model)
    base["simmodel_status_no_rec"], base["simmodel_rt_no_rec_d"] = bst, brt
    table = table.merge(base[[cid, "simmodel_status_no_rec", "simmodel_rt_no_rec_d"]], on=cid)

    d = lambda s: s.to_numpy(float) / SECONDS_PER_DAY
    f = lambda s: s.to_numpy(float)
    b = table.groupby(cid).first()
    rows = []
    # levels: recommendations
    rows += compare("outcome, recommendation", f(table["sim_status_method_mean"]),
                    f(table["pred_status_with_rec"]), f(table["simmodel_status_with_rec"]))
    rows += compare("time (d), recommendation", d(table["sim_remaining_time_method_mean"]),
                    d(table["pred_remaining_time_with_rec"]), f(table["simmodel_rt_with_rec_d"]))
    # levels: baseline (one per case)
    rows += compare("outcome, no recommendation", f(b["sim_status_baseline_mean"]),
                    f(b["pred_status_no_rec"]), f(b["simmodel_status_no_rec"]))
    rows += compare("time (d), no recommendation", d(b["sim_remaining_time_baseline_mean"]),
                    d(b["pred_remaining_time_no_rec"]), f(b["simmodel_rt_no_rec_d"]))
    # effect of the recommendation (rec - baseline)
    eff = lambda a, c: f(table[a]) - f(table[c])
    rows += compare("outcome effect (rec - no rec)", eff("sim_status_method_mean", "sim_status_baseline_mean"),
                    eff("pred_status_with_rec", "pred_status_no_rec"),
                    eff("simmodel_status_with_rec", "simmodel_status_no_rec"))
    rows += compare("time effect (d, rec - no rec)",
                    (f(table["sim_remaining_time_method_mean"]) - f(table["sim_remaining_time_baseline_mean"])) / SECONDS_PER_DAY,
                    (f(table["pred_remaining_time_with_rec"]) - f(table["pred_remaining_time_no_rec"])) / SECONDS_PER_DAY,
                    f(table["simmodel_rt_with_rec_d"]) - f(table["simmodel_rt_no_rec_d"]))
    res_df = pd.DataFrame(rows)

    # sign agreement of the effect: does the model predict an improvement where ProSiT shows one?
    sign_rows = []
    for name, simv, realv, simmv in (
            ("outcome effect", eff("sim_status_method_mean", "sim_status_baseline_mean"),
             eff("pred_status_with_rec", "pred_status_no_rec"), eff("simmodel_status_with_rec", "simmodel_status_no_rec")),
            ("time effect", eff("sim_remaining_time_method_mean", "sim_remaining_time_baseline_mean"),
             eff("pred_remaining_time_with_rec", "pred_remaining_time_no_rec"),
             f(table["simmodel_rt_with_rec_d"]) - f(table["simmodel_rt_no_rec_d"]))):
        nz = simv != 0
        sign_rows.append({"quantity": name, "rows with ProSiT effect != 0": int(nz.sum()),
                          "same sign, real model %": 100 * np.mean(np.sign(realv[nz]) == np.sign(simv[nz])),
                          "same sign, BPI12_sim model %": 100 * np.mean(np.sign(simmv[nz]) == np.sign(simv[nz]))})

    # rows where the recommended step barely moves the prediction away from the no-recommendation baseline
    near = lambda a, c, tol: 100 * float(np.mean(np.abs(f(table[a]) - f(table[c])) < tol))
    rt_real = (f(table["pred_remaining_time_with_rec"]) - f(table["pred_remaining_time_no_rec"])) / SECONDS_PER_DAY
    rt_sim = f(table["simmodel_rt_with_rec_d"]) - f(table["simmodel_rt_no_rec_d"])
    flat_rows = [{"model": "real model",
                  "|outcome - baseline| < 0.01 %": near("pred_status_with_rec", "pred_status_no_rec", 0.01),
                  "|time - baseline| < 0.1 d %": 100 * float(np.mean(np.abs(rt_real) < 0.1))},
                 {"model": "BPI12_sim model",
                  "|outcome - baseline| < 0.01 %": near("simmodel_status_with_rec", "simmodel_status_no_rec", 0.01),
                  "|time - baseline| < 0.1 d %": 100 * float(np.mean(np.abs(rt_sim) < 0.1))}]

    # noise of a mean over n_sim runs: outcome is binomial; E|p_hat - p| ~ sqrt(2/pi) * sqrt(p(1-p)/n)
    p = np.clip(f(table["pred_status_with_rec"]), 0, 1)
    noise_mae = float(np.mean(np.sqrt(2 / np.pi) * np.sqrt(p * (1 - p) / args.n_sim)))

    stats = pd.DataFrame([training_stats(args.case_study), training_stats(args.sim_case_study)])
    lines =["=" * 100,
             f"PREDICTED (real / {args.sim_case_study} models) vs SIMULATED (ProSiT) -- {args.case_study}, {args.method}, all test cases",
             f"Written by 8d_aggregate_pred_vs_sim.py - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 100,
             f"rows (case, rank): {len(table)} | cases: {table[cid].nunique()} | ProSiT: mean over {args.n_sim} runs",
             "outcome: predicted probability vs share of positive runs; time: remaining time in days.",
             "effect: recommendation minus no-recommendation baseline (same case).",
             "'real vs BPI12_sim model': the two predictions against each other (MAE, mean difference, r).", "",
             "(a) Training sets", stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"), "",
             "(b) Predictions against the simulation",
             res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), "",
             "(c) Sign of the effect of the recommendation (rows where ProSiT shows an effect)",
             pd.DataFrame(sign_rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"), "",
             "(c2) Rows (case, rank) where the recommended step barely changes the prediction of the no-recommendation",
             "     baseline (the model hardly uses the recommended step there)",
             pd.DataFrame(flat_rows).to_string(index=False, float_format=lambda x: f"{x:.1f}"), "",
             f"(d) Outcome noise of a {args.n_sim}-run mean: expected |share of positive runs - true probability| "
             f"~ {noise_mae:.3f} (binomial, true probability ~ real-model prediction): an outcome MAE of this size is "
             "reached even by a perfect model."]
    out = HERE / f"8d_pred_vs_sim_{args.case_study}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
