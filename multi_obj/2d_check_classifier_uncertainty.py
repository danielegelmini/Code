"""
Should the CLASSIFIER's uncertainty become a 4th Pareto objective?

The outcome classifier is now trained with posterior_sampling=True and exposes,
via UncertaintyClassifier.predict_uncertainty, the entropy decomposition
(Malinin et al., "Uncertainty in Gradient Boosting via Ensembles", ICLR 2021,
eq. 4):
    data / aleatoric  = E_m H(p_m)        -- expected entropy of the members
    total             = H( mean_m p_m )   -- entropy of the mean probability
    knowledge / epist. = total - data     -- mutual information (BALD)

Two things must hold for it to be worth a 4th objective; this script checks both
for one case study:

  1. NOT REDUNDANT WITH OBJECTIVE #1.  Objective #1 is "maximise P(y=1)". The
     aleatoric entropy of a Bernoulli is a deterministic function of P
     (H(0.5)=0.69, H(0.9)=0.33, ...), so "minimise total uncertainty" is largely
     "maximise |P - 0.5|", i.e. a mirror of objective #1. We measure:
       - how close data_entropy is to the exact binary entropy of P;
       - Spearman(P, total_entropy) and Spearman(P, knowledge_entropy) on the
         P > 0.5 side -- if total is strongly anti-correlated with P it adds no
         new trade-off; knowledge should be roughly independent of P.

  2. THE EPISTEMIC PART ACTUALLY DISCRIMINATES.  Same test as the (deleted)
     regressor check: does knowledge_entropy rise on rare / never-seen
     NEXT_RESOURCE and (activity -> NEXT_ACTIVITY) transitions -- the regime the
     Pareto search probes -- both on the real test rows and on synthetic
     candidates whose NEXT_RESOURCE is swapped to a rare / unseen value?

Verdict logic
-------------
- If total_entropy is ~ -corr with P AND knowledge_entropy is tiny / flat across
  rarity buckets -> a 4th objective adds a near-copy of objective #1 plus noise.
  Keep the classifier uncertainty as DIAGNOSTIC ONLY (option B).
- If knowledge_entropy clearly rises on rare / unseen inputs -> folding a
  pessimistic P_lcb = P - k * knowledge_std into objective #1 (option C, still
  3 objectives) is worth trying.

Run
---
  python 2d_check_classifier_uncertainty.py --case_study BPI12
  python 2d_check_classifier_uncertainty.py --case_study BAC --n_synth 400
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import joblib
from scipy.stats import spearmanr

from utils.get_features import get_features
from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.predictive_models_functions import prepare_df_for_ml, _binary_entropy

END_DATE_NAME = "time:timestamp"
START_DATE_NAME = "start:timestamp"
ACT = "concept:name"
NEXT_ACT = "NEXT_ACTIVITY"
NEXT_RES = "NEXT_RESOURCE"

FREQ_BUCKETS = [(-1, 0, "unseen"), (0, 10, "1-10"), (10, 100, "11-100"),
                (100, 1000, "101-1000"), (1000, np.inf, ">1000")]


def _uncertainty_frame(pipe, X):
    """predict_uncertainty through the loaded pipeline -> DataFrame."""
    step = pipe.named_steps["prediction"]
    Xt = pipe.named_steps["transformation"].transform(X)
    out = step.predict_uncertainty(Xt)
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in out.items()})


def _bucket(series_freq):
    labels = pd.Series(index=series_freq.index, dtype=object)
    for lo, hi, name in FREQ_BUCKETS:
        labels[(series_freq > lo) & (series_freq <= hi)] = name
    return labels


def _load(case_study):
    case_id_name, _, _, _, _, columns_to_remove = get_features(case_study)
    d = Path(f"./case_studies/{case_study}")
    train = pd.read_csv(d / "train_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    test = pd.read_csv(d / "test_data.csv", parse_dates=[END_DATE_NAME, START_DATE_NAME])
    if case_study == "BPI12":
        train = convert_dtypes_bpi12(train, "experiment")
        test = convert_dtypes_bpi12(test, "experiment")
    X_test, _, _ = prepare_df_for_ml(test, case_id_name, columns_to_remove)
    return train, test, X_test


def redundancy_check(unc):
    print("\n" + "=" * 74)
    print("REDUNDANCY WITH OBJECTIVE #1  (maximise P(y = positive))")
    print("=" * 74)
    p = unc["proba"].to_numpy()
    exact_h = _binary_entropy(p)
    print(f"  data_entropy vs exact binary entropy of P: max|diff| = {np.max(np.abs(unc['data_entropy'] - exact_h)):.4f}  "
          f"(they should be nearly identical -> data uncertainty carries no info beyond P)")

    conf = p > 0.5  # the side the Pareto search actually selects from
    for col in ["total_entropy", "data_entropy", "knowledge_entropy"]:
        rho_all = spearmanr(p, unc[col]).statistic
        rho_conf = spearmanr(p[conf], unc[col].to_numpy()[conf]).statistic if conf.sum() > 5 else np.nan
        print(f"  Spearman(P, {col:<18}) = {rho_all:+.3f}  |  on P>0.5 only = {rho_conf:+.3f}")
    print("  -> a strong negative Spearman on P>0.5 means the objective is a mirror of #1;")
    print("     knowledge_entropy near 0 means it is the only part independent of P.")


def rarity_check(case_study, pipe):
    train, test, X_test = _load(case_study)
    unc = _uncertainty_frame(pipe, X_test)

    res_freq = train[NEXT_RES].astype(str).value_counts()
    trans_freq = (train[ACT].astype(str) + " -> " + train[NEXT_ACT].astype(str)).value_counts()
    test_res_f = test[NEXT_RES].astype(str).map(res_freq).fillna(0)
    test_trans_f = (test[ACT].astype(str) + " -> " + test[NEXT_ACT].astype(str)).map(trans_freq).fillna(0)

    print("\n" + "=" * 74)
    print(f"RARITY CHECK -- does knowledge_entropy rise on rare inputs?  ({case_study}, {len(unc)} rows)")
    print("=" * 74)
    print(f"  Overall: data {unc['data_entropy'].mean():.5f} | knowledge {unc['knowledge_entropy'].mean():.5f} | "
          f"total {unc['total_entropy'].mean():.5f}  "
          f"(epistemic share {unc['knowledge_entropy'].mean() / max(unc['total_entropy'].mean(), 1e-12) * 100:.1f}%)")

    for name, freq in [("NEXT_RESOURCE", test_res_f), ("activity->NEXT_ACTIVITY", test_trans_f)]:
        print(f"\n  Binned by {name} training frequency:")
        g = unc.assign(bucket=_bucket(freq).values).groupby("bucket")
        rows = []
        for _, _, lbl in FREQ_BUCKETS:
            if lbl in g.groups:
                sub = g.get_group(lbl)
                rows.append((lbl, len(sub), sub["knowledge_entropy"].mean(), sub["data_entropy"].mean()))
        base = next((r[2] for r in rows if r[0] in (">1000", "101-1000")), rows[-1][2] if rows else np.nan)
        print(f"    {'bucket':<12} {'n':>7} {'knowledge ent':>14} {'data ent':>12} {'knowl vs common':>16}")
        for lbl, n, k, d in rows:
            print(f"    {lbl:<12} {n:>7} {k:>14.5f} {d:>12.5f} {(k / base if base else np.nan):>15.2f}x")
        mask = freq.values > 0
        rho = spearmanr(np.log(freq.values[mask] + 1), unc["knowledge_entropy"].values[mask]).statistic
        print(f"    Spearman(log freq, knowledge_entropy) = {rho:+.3f}  (want clearly negative)")


def synthetic_check(case_study, pipe, n_synth, seed):
    train, test, X_test = _load(case_study)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n_synth, len(X_test)), replace=False)
    base = X_test.iloc[idx].reset_index(drop=True)

    res_by_act = train.groupby(ACT)[NEXT_RES].apply(lambda s: s.astype(str).value_counts())
    res_global = train[NEXT_RES].astype(str).value_counts()
    rare_pool = res_global[res_global <= 5].index.tolist() or res_global.index[-20:].tolist()

    def variant(kind):
        df = base.copy()
        if kind == "real":
            return df
        new = []
        for act in df[ACT].astype(str):
            if kind == "rare":
                try:
                    counts = res_by_act.loc[act]
                    new.append(str(counts.index[-1]) if len(counts) else str(rng.choice(rare_pool)))
                except Exception:
                    new.append(str(rng.choice(rare_pool)))
            else:
                new.append("__UNSEEN_RESOURCE__")
        df[NEXT_RES] = new
        return df

    print("\n" + "=" * 74)
    print(f"SYNTHETIC CHECK -- swap NEXT_RESOURCE  ({case_study}, {len(base)} query instances)")
    print("=" * 74)
    print(f"    {'NEXT_RESOURCE set to':<28} {'knowledge ent':>14} {'data ent':>12} {'mean P':>8}")
    ref = None
    for kind, label in [("real", "its real value"), ("rare", "rarest for that activity"),
                        ("unseen", "a string never in training")]:
        u = _uncertainty_frame(pipe, variant(kind))
        k, d, mp = u["knowledge_entropy"].mean(), u["data_entropy"].mean(), u["proba"].mean()
        ref = k if ref is None else ref
        print(f"    {label:<28} {k:>14.5f} {d:>12.5f} {mp:>8.3f}   ({k / ref if ref else np.nan:.2f}x vs real)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case_study", default="BPI12")
    ap.add_argument("--n_synth", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = Path(f"./case_studies/{args.case_study}/model/catboost_model_label.joblib")
    pipe = joblib.load(model_path)
    step = pipe.named_steps["prediction"]
    if not hasattr(step, "predict_uncertainty"):
        raise SystemExit(f"{model_path} has no uncertainty support -- retrain with posterior_sampling=True first.")
    print(f"Loaded {model_path}\n  wrapper={type(step).__name__}  temperature={getattr(step, 'temperature', 1.0):.3f}  "
          f"trees={step.fitted_model.tree_count_}")

    train, test, X_test = _load(args.case_study)
    redundancy_check(_uncertainty_frame(pipe, X_test))
    rarity_check(args.case_study, pipe)
    synthetic_check(args.case_study, pipe, args.n_synth, args.seed)

    print("\n" + "=" * 74)
    print("HOW TO READ IT")
    print("-" * 74)
    print(
        "4th objective is worth it ONLY if knowledge_entropy rises clearly (>1.5x) on\n"
        "the unseen / 1-10 buckets and the synthetic 'unseen' resource, AND is roughly\n"
        "independent of P (small Spearman(P, knowledge_entropy)). If total_entropy just\n"
        "mirrors P (strong negative Spearman on P>0.5) and knowledge stays flat, keep\n"
        "the classifier uncertainty as diagnostic only -- a 4th objective would add a\n"
        "near-copy of objective #1 plus noise."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
