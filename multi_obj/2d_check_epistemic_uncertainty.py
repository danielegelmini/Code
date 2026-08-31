"""
Does the regressor's EPISTEMIC (knowledge) uncertainty actually do its job?

The time model is trained with RMSEWithUncertainty + posterior_sampling (SGLB),
which is slower and constrains the hyperparameter space, and whose only payoff
is the epistemic component of predict_uncertainty(). In the training report the
epistemic share of the variance is tiny (~0.2%). That average, though, is
dominated by ordinary test cases. The question that matters is whether the
epistemic std RISES on the inputs the Pareto search actually probes: rare or
never-seen (NEXT_ACTIVITY, NEXT_RESOURCE) combinations.

This script checks exactly that, two ways, for one case study:

  1. NATURAL cut -- bucket the real test rows by how often their NEXT_RESOURCE
     and their (activity -> NEXT_ACTIVITY) transition were seen in training, and
     compare mean epistemic std across buckets. Also the Spearman correlation
     between epistemic std and -log(frequency).

  2. SYNTHETIC cut -- take a sample of query instances and score each with:
       (a) its real (NEXT_ACTIVITY, NEXT_RESOURCE),
       (b) a plausible-but-rare resource for that activity,
       (c) a resource string that never appears in training at all.
     If epistemic uncertainty is meaningful, (c) >> (b) >= (a).

Verdict: if epistemic std is essentially flat across all of these, SGLB /
posterior_sampling is not buying anything and the model could drop to plain
aleatoric-only uncertainty. If it rises clearly on the rare / unseen buckets,
it is worth keeping.

Run
---
  python 2d_check_epistemic_uncertainty.py --case_study BPI12
  python 2d_check_epistemic_uncertainty.py --case_study BAC --n_synth 400
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr

from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.predictive_models_functions import prepare_df_for_ml
from utils.recommendation_functions import predict_time_and_uncertainty
import joblib

END_DATE_NAME = "time:timestamp"
START_DATE_NAME = "start:timestamp"
ACT = "concept:name"
NEXT_ACT = "NEXT_ACTIVITY"
NEXT_RES = "NEXT_RESOURCE"

FREQ_BUCKETS = [(-1, 0, "unseen"), (0, 10, "1-10"), (10, 100, "11-100"),
                (100, 1000, "101-1000"), (1000, np.inf, ">1000")]


def _uncertainty_frame(pipe, X):
    """predict_uncertainty through the loaded pipeline -> DataFrame of stds."""
    step = pipe.named_steps["prediction"]
    Xt = pipe.named_steps["transformation"].transform(X)
    out = step.predict_uncertainty(Xt)
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in out.items()})


def _bucket(series_freq):
    labels = pd.Series(index=series_freq.index, dtype=object)
    for lo, hi, name in FREQ_BUCKETS:
        labels[(series_freq > lo) & (series_freq <= hi)] = name
    return labels


def natural_cut(case_study, pipe):
    case_id_name, _, _, _, _, columns_to_remove = get_features(case_study)
    d = Path(f"./case_studies/{case_study}")
    train = pd.read_csv(d / "train_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    test = pd.read_csv(d / "test_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    if case_study == "BPI12":
        train = convert_dtypes_bpi12(train, "experiment")
        test = convert_dtypes_bpi12(test, "experiment")

    X_test, _, _ = prepare_df_for_ml(test, case_id_name, columns_to_remove)
    unc = _uncertainty_frame(pipe, X_test)

    res_freq = train[NEXT_RES].astype(str).value_counts()
    trans_freq = (train[ACT].astype(str) + " -> " + train[NEXT_ACT].astype(str)).value_counts()

    test_res_f = test[NEXT_RES].astype(str).map(res_freq).fillna(0)
    test_trans_f = (test[ACT].astype(str) + " -> " + test[NEXT_ACT].astype(str)).map(trans_freq).fillna(0)

    print("\n" + "=" * 74)
    print(f"NATURAL CUT  --  {case_study}  ({len(unc)} test rows)")
    print("=" * 74)
    print(f"Overall: epistemic std mean={unc['knowledge_std'].mean():.5f}  "
          f"aleatoric std mean={unc['data_std'].mean():.5f}  "
          f"epistemic/total var share={ (unc['knowledge_std']**2).mean() / (unc['total_std']**2).mean() * 100:.2f}%")

    for name, freq in [("NEXT_RESOURCE", test_res_f), ("activity->NEXT_ACTIVITY transition", test_trans_f)]:
        print(f"\n  Binned by {name} training frequency:")
        buck = _bucket(freq)
        g = unc.assign(bucket=buck.values).groupby("bucket")
        rows = []
        for lo, hi, lbl in FREQ_BUCKETS:
            if lbl not in g.groups:
                continue
            sub = g.get_group(lbl)
            rows.append((lbl, len(sub), sub["knowledge_std"].mean(), sub["data_std"].mean()))
        base_epi = next((r[2] for r in rows if r[0] in (">1000", "101-1000")), rows[-1][2] if rows else np.nan)
        print(f"    {'bucket':<12} {'n':>7} {'epistemic std':>14} {'aleatoric std':>14} {'epi vs common':>14}")
        for lbl, n, epi, ale in rows:
            ratio = epi / base_epi if base_epi else np.nan
            print(f"    {lbl:<12} {n:>7} {epi:>14.5f} {ale:>14.5f} {ratio:>13.2f}x")

        mask = freq.values > 0
        rho = spearmanr(np.log(freq.values[mask] + 1), unc["knowledge_std"].values[mask]).statistic
        print(f"    Spearman(log freq, epistemic std) = {rho:+.3f}   "
              f"(want clearly negative: rarer -> more epistemic)")


def synthetic_cut(case_study, pipe, n_synth, seed):
    case_id_name, _, _, _, _, columns_to_remove = get_features(case_study)
    d = Path(f"./case_studies/{case_study}")
    train = pd.read_csv(d / "train_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    test = pd.read_csv(d / "test_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    if case_study == "BPI12":
        train = convert_dtypes_bpi12(train, "experiment")
        test = convert_dtypes_bpi12(test, "experiment")

    X_test, _, _ = prepare_df_for_ml(test, case_id_name, columns_to_remove)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n_synth, len(X_test)), replace=False)
    base = X_test.iloc[idx].reset_index(drop=True)

    # resources per activity, ranked by frequency
    res_by_act = (
        train.groupby(ACT)[NEXT_RES].apply(lambda s: s.astype(str).value_counts())
    )
    res_global = train[NEXT_RES].astype(str).value_counts()
    rare_res_pool = res_global[res_global <= 5].index.tolist() or res_global.index[-20:].tolist()

    def variant(kind):
        df = base.copy()
        new_res = []
        for act in df[ACT].astype(str):
            if kind == "real":
                new_res.append(None)  # keep
            elif kind == "rare":
                try:
                    counts = res_by_act.loc[act]
                    cand = counts.index[-1] if len(counts) else rng.choice(rare_res_pool)
                except Exception:
                    cand = rng.choice(rare_res_pool)
                new_res.append(str(cand))
            else:  # unseen
                new_res.append("__UNSEEN_RESOURCE__")
        if kind != "real":
            df[NEXT_RES] = new_res
        return df

    print("\n" + "=" * 74)
    print(f"SYNTHETIC CUT  --  {case_study}  ({len(base)} sampled query instances)")
    print("=" * 74)
    print(f"    {'NEXT_RESOURCE set to':<28} {'epistemic std':>14} {'aleatoric std':>14}")
    ref = None
    for kind, label in [("real", "its real value"), ("rare", "rarest for that activity"),
                        ("unseen", "a string never in training")]:
        u = _uncertainty_frame(pipe, variant(kind))
        epi, ale = u["knowledge_std"].mean(), u["data_std"].mean()
        if ref is None:
            ref = epi
        print(f"    {label:<28} {epi:>14.5f} {ale:>14.5f}   ({epi / ref:.2f}x vs real)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case_study", default="BPI12")
    ap.add_argument("--n_synth", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = Path(f"./case_studies/{args.case_study}/model/catboost_model_sigmoid_mm.joblib")
    pipe = joblib.load(model_path)
    step = pipe.named_steps["prediction"]
    if not hasattr(step, "predict_uncertainty"):
        raise SystemExit(f"{model_path} has no uncertainty support -- retrain with RMSEWithUncertainty first.")
    print(f"Loaded {model_path}\n  wrapper={type(step).__name__}  sigma_scale={getattr(step, 'sigma_scale', 1.0):.3f}  "
          f"trees={step.fitted_model.tree_count_}  virtual_ensembles={step.virtual_ensembles_count}")

    natural_cut(args.case_study, pipe)
    synthetic_cut(args.case_study, pipe, args.n_synth, args.seed)

    print("\n" + "=" * 74)
    print("HOW TO READ IT")
    print("-" * 74)
    print(
        "Epistemic uncertainty is worth the SGLB cost only if 'epistemic std' rises\n"
        "clearly (say >1.5x) on the unseen / 1-10 buckets and on the synthetic\n"
        "'unseen' resource, and the Spearman correlation is solidly negative. If it\n"
        "stays within ~10-20% of the common-case value everywhere, posterior_sampling\n"
        "is not earning its keep and the model can use aleatoric-only uncertainty."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
