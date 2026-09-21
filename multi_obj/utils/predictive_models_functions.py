import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
from optuna.integration import CatBoostPruningCallback
from scipy.optimize import minimize_scalar

from utils.train_test_split import extract_internal_running_validation

DEFAULT_VIRTUAL_ENSEMBLES_COUNT = 10
DEFAULT_CALIB_BINS = 15

# Classification
def _expected_calibration_error(y_true, proba, n_bins=10):
    """Computes the Expected Calibration Error (ECE): the average, over equal-width
    probability bins, of |mean predicted probability - observed positive rate| in
    that bin, weighted by the bin's share of examples.

    Args:
        y_true (array-like of int): True binary labels (0/1).
        proba (array-like of float): Predicted probability of the positive class.
        n_bins (int, optional): Number of equal-width bins over [0, 1]. Defaults to 10.

    Returns:
        float: The ECE, in [0, 1] -- lower means better calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba > lo) & (proba <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)

def _fit_temperature(logits, y_true):
    """Fits a single-scalar temperature T that minimises the Bernoulli log-loss of
    sigmoid(logit / T) on a held-out slice (Guo et al. 2017 temperature scaling).
    T > 1 softens over-confident probabilities, T < 1 sharpens them.

    Args:
        logits (array-like of float): Raw (pre-sigmoid) model outputs on a held-out slice.
        y_true (array-like of int): True binary labels (0/1) for the same slice.

    Returns:
        float: The fitted temperature T, clipped to [0.2, 5.0].
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    def nll(temp):
        """Bernoulli negative log-likelihood of the slice under a candidate temperature.

        Args:
            temp (float): Candidate temperature.

        Returns:
            float: Mean negative log-likelihood over the slice.
        """
        p = np.clip(1.0 / (1.0 + np.exp(-logits / temp)), 1e-7, 1.0 - 1e-7)
        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(np.clip(res.x, 0.2, 5.0))

# Regression
def _fit_sigma_scale(resid, sigma):
    """Fits a single-scalar std-scaling factor s that minimises the Gaussian negative
    log-likelihood of N(mu, (s * sigma)^2) on a held-out slice -- the regression
    analogue of temperature scaling (Levi et al. 2022, eq. 11-12). s > 1 inflates
    intervals that were too tight, s < 1 shrinks intervals that were too wide.
    The exact minimiser is sqrt(mean(resid^2 / sigma^2)); it is obtained here by
    a bounded 1-D search, and the predicted variance is floored, so a stray
    near-zero variance from RMSEWithUncertainty cannot drive the estimate to the
    bound.

    Args:
        resid (array-like of float): True value minus predicted mean, on a held-out slice.
        sigma (array-like of float): Predicted standard deviation for the same slice.

    Returns:
        float: The fitted scale factor s, clipped to [0.2, 5.0].
    """
    resid = np.asarray(resid, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    var = np.clip(sigma ** 2, 1e-12, None)
    var = np.clip(var, 1e-4 * float(np.mean(var)), None)

    def nll(s):
        """Gaussian negative log-likelihood of the residuals under a candidate scale.

        Args:
            s (float): Candidate std-scaling factor.

        Returns:
            float: Mean negative log-likelihood over the slice (additive constant dropped).
        """
        v = (s ** 2) * var
        return float(np.mean(0.5 * np.log(v) + resid ** 2 / (2.0 * v)))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(np.clip(res.x, 0.2, 5.0))

def _regression_calibration_metrics(resid, sigma, n_bins=DEFAULT_CALIB_BINS):
    """Levi et al. (2022) calibration diagnostics for a Gaussian regression
    forecaster. Examples are sorted by predicted std and split into n_bins
    equal-count bins; per bin j,

        RMV(j)  = sqrt( mean_{t in B_j} sigma_t^2 )         (eq. 6)
        RMSE(j) = sqrt( mean_{t in B_j} (y_t - mu_t)^2 )    (eq. 7)

    A calibrated model has RMV(j) ~ RMSE(j) in every bin. The scalar summary is
    the expected normalized calibration error (eq. 8), the ENCE, the analogue of
    the classifier's ECE,

        ENCE = (1 / N) sum_j |RMV(j) - RMSE(j)| / RMV(j) ,

    reported next to the coefficient of variation of the predicted stds (eq. 9),
    cv = std(sigma) / mean(sigma), which must be well above zero for the
    uncertainty to carry per-example information (a constant sigma can reach
    ENCE ~ 0 while saying nothing). cv is invariant to the uniform sigma scaling,
    so raw and recalibrated stds share one value.

    Args:
        resid (array-like of float): True value minus predicted mean, per example.
        sigma (array-like of float): Predicted standard deviation, per example.
        n_bins (int, optional): Number of equal-COUNT bins to sort examples into
            (by predicted sigma). Defaults to DEFAULT_CALIB_BINS (15); clamped to
            the number of examples if fewer are given.

    Returns:
        dict: {
            "ence" (float): The scalar ENCE summary (eq. 8), lower is better.
            "cv" (float): Coefficient of variation of the predicted sigmas (eq. 9).
            "rmv_bins" (list of float): Per-bin RMV (predicted), low-sigma bin first.
            "rmse_bins" (list of float): Per-bin RMSE (observed), same bin order.
            "n_bins" (int): The actual number of bins used.
        }
    """
    resid = np.asarray(resid, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.size
    n_bins = int(max(1, min(n_bins, n)))
    order = np.argsort(sigma, kind="stable")
    resid, sigma = resid[order], sigma[order]

    rmv, rmse = [], []
    for b in np.array_split(np.arange(n), n_bins):
        rmv.append(float(np.sqrt(np.mean(sigma[b] ** 2))))
        rmse.append(float(np.sqrt(np.mean(resid[b] ** 2))))
    rmv = np.asarray(rmv)
    rmse = np.asarray(rmse)
    ence = float(np.mean(np.abs(rmv - rmse) / np.clip(rmv, 1e-12, None)))

    mean_sigma = float(np.mean(sigma))
    cv = float(np.std(sigma, ddof=1) / mean_sigma) if (n > 1 and mean_sigma > 0) else 0.0
    return {
        "ence": ence,
        "cv": cv,
        "rmv_bins": rmv.tolist(),
        "rmse_bins": rmse.tolist(),
        "n_bins": n_bins,
    }


class UncertaintyRegressor:
    """
    Wraps a CatBoost regressor trained with loss_function='RMSEWithUncertainty'
    and posterior_sampling=True (SGLB) so it can sit as the final step of an
    sklearn Pipeline while exposing the full predictive-uncertainty decomposition
    (Malinin et al., "Uncertainty in Gradient Boosting via Ensembles", ICLR 2021).

    - predict(X) returns only the mean prediction (column 0 of CatBoost's output),
      so every caller expecting a plain 1-D array of 'sigmoid_mm' values keeps
      working unchanged.
    - predict_uncertainty(X) runs the virtual ensemble and returns, via the law
      of total variance:
        data_std       = sqrt( mean_m sigma_m^2 )            (aleatoric)
        knowledge_std  = sqrt( Var_m mu_m )                  (epistemic)
        total_std      = sqrt( data_var + knowledge_var )
      plus `std` as an alias of total_std for backward compatibility. NB: on
      one-hot-encoded tabular data the epistemic part is typically tiny (the
      paper's own finding, and our 2d check) -- it is useful mainly to flag
      out-of-domain / rarely-seen inputs, not to improve error estimates.

    `sigma_scale` is a single post-hoc recalibration factor (>1 inflates,
    <1 shrinks), fitted on a held-out slice at training time. It rescales every
    std component by the same amount, so it does not change any ranking of cases
    by uncertainty; only the interval width moves.

    The target is NOT transformed here: 'sigmoid_mm' is already bounded in
    [0, 1] and an ablation showed a log1p transform did not help.
    """

    def __init__(self, fitted_model, sigma_scale=1.0, virtual_ensembles_count=DEFAULT_VIRTUAL_ENSEMBLES_COUNT):
        """Stores the already-fitted CatBoost regressor and its post-hoc calibration factor.

        Args:
            fitted_model (catboost.CatBoostRegressor): A model already trained with
                loss_function='RMSEWithUncertainty' and posterior_sampling=True.
            sigma_scale (float, optional): Post-hoc std recalibration factor (see
                _fit_sigma_scale). Defaults to 1.0 (no recalibration).
            virtual_ensembles_count (int, optional): Requested number of virtual
                ensemble members for uncertainty estimation. Defaults to
                DEFAULT_VIRTUAL_ENSEMBLES_COUNT (10); the number actually used may
                be lower, see _ve_count.
        """
        self.fitted_model = fitted_model
        self.sigma_scale = float(sigma_scale)
        self.virtual_ensembles_count = int(virtual_ensembles_count)

    def fit(self, X, y=None):
        """No-op fit, kept only so this wrapper satisfies scikit-learn's Pipeline/
        Estimator interface -- the wrapped model is already trained.

        Args:
            X: Ignored.
            y: Ignored. Defaults to None.

        Returns:
            UncertaintyRegressor: self, unchanged.
        """
        return self

    def __sklearn_is_fitted__(self):
        """Tells scikit-learn whether this estimator is ready to predict.

        Returns:
            bool: True if a wrapped model is present.
        """
        return self.fitted_model is not None

    @staticmethod
    def _two_col(preds):
        """Ensures a CatBoost prediction array has 2 columns (mean, variance).

        Args:
            preds (array-like): CatBoost's raw predict() output; 1-D or 2-D.

        Returns:
            numpy.ndarray: `preds` unchanged if already 2-D, otherwise `preds`
            stacked with a same-shaped column of zeros as column 1.
        """
        preds = np.asarray(preds, dtype=float)
        return preds if preds.ndim == 2 else np.column_stack([preds, np.zeros_like(preds)])

    def _ve_count(self):
        """Computes how many virtual ensemble members can safely be requested.

        CatBoost's virtual_ensembles_predict splits the model's trees into
        virtual_ensembles_count contiguous groups and needs at least 2 trees per
        group (tree_count_ >= 2 * count), not just tree_count_ >= count as a naive
        clamp would assume -- it raises "Not enough trees in model for N virtual
        Ensembles" otherwise. This matters because the final model's tree count is
        best_iteration + 1 from the winning Optuna trial (see train_ml_model),
        which can be a single-digit number when a trial's validation loss stops
        improving almost immediately.

        Returns:
            int: min(self.virtual_ensembles_count, tree_count_ // 2), floored at 0.
        """
        n_trees = getattr(self.fitted_model, "tree_count_", 0) or 0
        return int(max(0, min(self.virtual_ensembles_count, n_trees // 2)))

    def predict(self, X):
        """Point prediction only (no uncertainty).

        Args:
            X: Feature matrix, already preprocessed (as produced by the Pipeline's
                earlier steps).

        Returns:
            numpy.ndarray: 1-D array of predicted 'sigmoid_mm' means, one per row of X.
        """
        return self._two_col(self.fitted_model.predict(X))[:, 0] #return the mean

    def predict_uncertainty(self, X):
        """Predicts the mean and its aleatoric/epistemic/total uncertainty via the
        virtual-ensemble decomposition (law of total variance).

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            dict: {
                "mean" (numpy.ndarray): Predicted 'sigmoid_mm' mean, one per row of X.
                "data_std" (numpy.ndarray): Aleatoric (data-noise) component of std.
                "knowledge_std" (numpy.ndarray): Epistemic (model-knowledge) component.
                "total_std" (numpy.ndarray): sqrt(data_var + knowledge_var).
                "std" (numpy.ndarray): Alias of total_std, kept for backward compatibility.
            }
            Every std component is scaled by self.sigma_scale. If the model has too
            few trees for even one virtual ensemble, all std arrays are returned as
            zeros (with a warning printed) instead of raising.
        """
        ve_count = self._ve_count()
        if ve_count < 1:
            # Fewer than 2 trees total -- can't form even one virtual ensemble. Degrade to a zero-uncertainty point prediction rather than crashing
            n_trees = getattr(self.fitted_model, "tree_count_", 0) or 0
            print(f"WARNING: final model has only {n_trees} tree(s) -- too few for "
                  f"virtual-ensemble uncertainty. Reporting zero std.")
            mean = self.predict(X)
            zeros = np.zeros_like(mean)
            return {"mean": mean, "data_std": zeros, "knowledge_std": zeros,
                    "total_std": zeros, "std": zeros}

        # mean, knowledge var, data var
        out = np.asarray(
            self.fitted_model.virtual_ensembles_predict(
                X, prediction_type="TotalUncertainty",
                virtual_ensembles_count=ve_count,
            ),
            dtype=float,
        )
        mean = out[:, 0]
        knowledge_var = np.clip(out[:, 1], 0.0, None)
        data_var = np.clip(out[:, 2], 0.0, None)
        s = self.sigma_scale
        total_std = s * np.sqrt(data_var + knowledge_var)
        return {
            "mean": mean,
            "data_std": s * np.sqrt(data_var),
            "knowledge_std": s * np.sqrt(knowledge_var),
            "total_std": total_std,
            "std": total_std,
        }

    def predict_members(self, X):
        """Returns each virtual-ensemble member's predicted mean separately (not
        aggregated), so two rows can be compared member-by-member elsewhere.

        Used by the confidence-as-KPI filter (see
        utils/recommendation_functions.py) to compare a baseline row and a
        candidate row member-by-member: since both come from this same
        fitted model, comparing them on shared member indices captures their
        correlation for free, with no independence assumption and no normal
        approximation.

        CatBoost's prediction_type="VirtEnsembles" (not "VirtualEnsembles")
        returns, for regression, one (mean, var) pair per member; only the
        per-member mean is needed here, so the variance column is dropped.

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            numpy.ndarray: Shape (n_rows, ve_count) -- one column per virtual
            ensemble member, each holding that member's predicted mean. Degrades
            to shape (n_rows, 1) (the plain point prediction, repeated) when
            there are too few trees for even one virtual ensemble (mirrors
            predict_uncertainty()'s degradation for that case).
        """
        ve_count = self._ve_count()
        if ve_count < 1:
            mean = self.predict(X)
            return mean.reshape(-1, 1)

        out = np.asarray(
            self.fitted_model.virtual_ensembles_predict(
                X, prediction_type="VirtEnsembles",
                virtual_ensembles_count=ve_count,
            ),
            dtype=float,
        ) # out => (n_row, ve_count, 2), the third dimension is mean and variance
        return out[:, :, 0] #mean for all row of X for all  ve

    def get_params(self, deep=True):
        """Exposes this wrapper's constructor arguments, scikit-learn style.

        Args:
            deep (bool, optional): Ignored (no nested sub-estimators to recurse
                into). Present only for interface compatibility. Defaults to True.

        Returns:
            dict: {"fitted_model", "sigma_scale", "virtual_ensembles_count"}.
        """
        return {
            "fitted_model": self.fitted_model,
            "sigma_scale": self.sigma_scale,
            "virtual_ensembles_count": self.virtual_ensembles_count,
        }

    def set_params(self, **params):
        """Sets one or more constructor arguments after the fact, scikit-learn style.

        Args:
            **params: Any subset of {"fitted_model", "sigma_scale",
                "virtual_ensembles_count"} (or "temperature" for
                UncertaintyClassifier), as keyword arguments.

        Returns:
            The wrapper instance: self, mutated in place.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class UncertaintyClassifier:
    """
    Wraps a CatBoost classifier trained with posterior_sampling=True (SGLB) so
    the Pipeline can expose, besides the usual probability, the ensemble-based
    uncertainty decomposition for classification (Malinin et al., ICLR 2021):
        data / aleatoric  = E_m H(p_m)                (expected entropy of members)
        total             = H( mean_m p_m )           (entropy of the mean prob)
        knowledge / epist. = total - data             (mutual information / BALD)
    All in nats. Knowledge uncertainty is what an ensemble adds over a single
    model; on this data it is small in magnitude but is the signal for
    out-of-domain / anomalous inputs.

    predict() and predict_proba() are unchanged (raw CatBoost outputs), so
    nothing downstream shifts. `temperature` is a single post-hoc calibration
    scalar (Guo et al. 2017 temperature scaling), fitted on a held-out slice:
    p_cal = sigmoid( logit(p) / T ), T > 1 softening over-confident probabilities.
    It is returned by predict_uncertainty()["proba_calibrated"] and is NOT
    applied by predict_proba() -- switching the recommendation pipeline to the
    calibrated probability is a separate, deliberate step.
    """

    def __init__(self, fitted_model, temperature=1.0,
                 virtual_ensembles_count=DEFAULT_VIRTUAL_ENSEMBLES_COUNT):
        """Stores the already-fitted CatBoost classifier and its calibration temperature.

        Args:
            fitted_model (catboost.CatBoostClassifier): A model already trained
                with posterior_sampling=True.
            temperature (float, optional): Post-hoc calibration temperature (see
                _fit_temperature). Defaults to 1.0 (no recalibration).
            virtual_ensembles_count (int, optional): Requested number of virtual
                ensemble members for uncertainty estimation. Defaults to
                DEFAULT_VIRTUAL_ENSEMBLES_COUNT (10); the number actually used may
                be lower, see _ve_count.
        """
        self.fitted_model = fitted_model
        self.temperature = float(temperature)
        self.virtual_ensembles_count = int(virtual_ensembles_count)

    def fit(self, X, y=None):
        """No-op fit, kept only so this wrapper satisfies scikit-learn's Pipeline/
        Estimator interface -- the wrapped model is already trained.

        Args:
            X: Ignored.
            y: Ignored. Defaults to None.

        Returns:
            UncertaintyClassifier: self, unchanged.
        """
        return self

    def __sklearn_is_fitted__(self):
        """Tells scikit-learn whether this estimator is ready to predict.

        Returns:
            bool: True if a wrapped model is present.
        """
        return self.fitted_model is not None

    @property
    def classes_(self):
        """The class labels the wrapped model was trained on (e.g. [0, 1]).

        Returns:
            numpy.ndarray: Delegates to the wrapped model's own `classes_`.
        """
        return self.fitted_model.classes_

    def _ve_count(self):
        """Computes how many virtual ensemble members can safely be requested.

        See UncertaintyRegressor._ve_count() for the full explanation: CatBoost's
        virtual_ensembles_predict needs at least 2 trees per requested member
        (tree_count_ >= 2 * count), not just tree_count_ >= count.

        Returns:
            int: min(self.virtual_ensembles_count, tree_count_ // 2), floored at 0.
        """
        n_trees = getattr(self.fitted_model, "tree_count_", 0) or 0
        return int(max(0, min(self.virtual_ensembles_count, n_trees // 2)))

    def predict(self, X):
        """Predicted class label, unchanged from the wrapped CatBoost model.

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            numpy.ndarray: Predicted class label per row of X.
        """
        return self.fitted_model.predict(X)

    def predict_proba(self, X):
        """Predicted class probabilities, RAW (not temperature-calibrated) --
        unchanged from the wrapped CatBoost model. Use predict_uncertainty()
        instead for the calibrated probability.

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            numpy.ndarray: Shape (n_rows, 2), columns [P(class 0), P(class 1)].
        """
        return self.fitted_model.predict_proba(X)

    def predict_uncertainty(self, X):
        """Predicts P(positive class), its temperature-calibrated version, and the
        aleatoric/epistemic/total entropy decomposition via the virtual ensemble.

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            dict: {
                "proba" (numpy.ndarray): Raw P(y=1), one per row of X.
                "proba_calibrated" (numpy.ndarray): sigmoid(logit(proba) / temperature).
                "data_entropy" (numpy.ndarray): Aleatoric entropy (expected entropy
                    of the ensemble members), in nats.
                "knowledge_entropy" (numpy.ndarray): Epistemic entropy (total -
                    data, i.e. mutual information / BALD), in nats.
                "total_entropy" (numpy.ndarray): Entropy of the mean member
                    probability, in nats.
            }
            If the model has too few trees for even one virtual ensemble, the
            three entropy arrays are returned as zeros (with a warning printed)
            instead of raising; "proba"/"proba_calibrated" are still computed.
        """
        proba = np.asarray(self.fitted_model.predict_proba(X), dtype=float)[:, 1]
        p = np.clip(proba, 1e-7, 1.0 - 1e-7)
        logit = np.log(p / (1.0 - p))
        proba_cal = 1.0 / (1.0 + np.exp(-logit / self.temperature)) #calibration

        ve_count = self._ve_count()
        if ve_count < 1:
            n_trees = getattr(self.fitted_model, "tree_count_", 0) or 0
            print(f"WARNING: final model has only {n_trees} tree(s) -- too few for "
                  f"virtual-ensemble uncertainty. Reporting zero entropy.")
            zeros = np.zeros_like(proba)
            return {
                "proba": proba, "proba_calibrated": proba_cal,
                "data_entropy": zeros, "knowledge_entropy": zeros, "total_entropy": zeros,
            }

        out = np.asarray(
            self.fitted_model.virtual_ensembles_predict(
                X, prediction_type="TotalUncertainty",
                virtual_ensembles_count=ve_count,
            ),
            dtype=float,
        )
        # classification: column 0 = data (expected) entropy, column 1 = total entropy
        data_entropy = np.clip(out[:, 0], 0.0, None)
        total_entropy = np.clip(out[:, 1], 0.0, None)
        knowledge_entropy = np.clip(total_entropy - data_entropy, 0.0, None)

        return {
            "proba": proba,
            "proba_calibrated": proba_cal,
            "data_entropy": data_entropy,
            "knowledge_entropy": knowledge_entropy,
            "total_entropy": total_entropy,
        }

    def predict_members(self, X):
        """Returns each virtual-ensemble member's CALIBRATED P(y=positive)
        separately (not aggregated), so two rows can be compared member-by-member
        elsewhere.

        Used by the confidence-as-KPI filter (see
        utils/recommendation_functions.py) to compare a baseline row and a
        candidate row member-by-member, the same way
        UncertaintyRegressor.predict_members() does for the time model.

        CatBoost's prediction_type="VirtEnsembles" returns, for
        classification, one RAW LOGIT per member (verified empirically:
        sigmoid(member_logit) matches predict_proba()'s raw probability) --
        not a probability. Applying the temperature transform directly to
        that logit, sigmoid(logit / T), gives the same per-member calibrated
        probability predict_uncertainty()["proba_calibrated"] would give for
        a single-member "ensemble", without needing to round-trip through
        log-odds of an already-clipped probability.

        Args:
            X: Feature matrix, already preprocessed.

        Returns:
            numpy.ndarray: Shape (n_rows, ve_count) -- one column per virtual
            ensemble member, each holding that member's temperature-calibrated
            P(y=1). Degrades to shape (n_rows, 1) (the plain calibrated point
            prediction, repeated) when there are too few trees for even one
            virtual ensemble (mirrors predict_uncertainty()'s degradation).
        """
        ve_count = self._ve_count()
        if ve_count < 1:
            proba = np.asarray(self.fitted_model.predict_proba(X), dtype=float)[:, 1]
            p = np.clip(proba, 1e-7, 1.0 - 1e-7)
            logit = np.log(p / (1.0 - p))
            proba_cal = 1.0 / (1.0 + np.exp(-logit / self.temperature))
            return proba_cal.reshape(-1, 1)

        out = np.asarray(
            self.fitted_model.virtual_ensembles_predict(
                X, prediction_type="VirtEnsembles",
                virtual_ensembles_count=ve_count,
            ),
            dtype=float,
        )
        member_logits = out[:, :, 0]
        return 1.0 / (1.0 + np.exp(-member_logits / self.temperature))

    def get_params(self, deep=True):
        """Exposes this wrapper's constructor arguments, scikit-learn style.

        Args:
            deep (bool, optional): Ignored (no nested sub-estimators to recurse
                into). Present only for interface compatibility. Defaults to True.

        Returns:
            dict: {"fitted_model", "temperature", "virtual_ensembles_count"}.
        """
        return {
            "fitted_model": self.fitted_model,
            "temperature": self.temperature,
            "virtual_ensembles_count": self.virtual_ensembles_count,
        }

    def set_params(self, **params):
        """Sets one or more constructor arguments after the fact, scikit-learn style.

        Args:
            **params: Any subset of {"fitted_model", "temperature",
                "virtual_ensembles_count"}, as keyword arguments.

        Returns:
            UncertaintyClassifier: self, mutated in place.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

def prepare_df_for_ml(df, case_id_name, columns_to_remove=None):
    """
    Prepares the dataframe for machine learning by separating features and targets.

    This function extracts the targets 'label' and 'sigmoid_mm', drops the specified case ID column, 
    and optionally removes other specified columns.

    Args:
        df (pandas.DataFrame): The input dataframe to process.
        case_id_name (str): The name of the column containing the case identifier to drop.
        columns_to_remove (list of str, optional): A list of additional column names to drop. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The feature matrix (X).
            - pandas.Series: The 'label' target variable (y1).
            - pandas.Series: The 'sigmoid_mm' target variable (y2).
    """
    df = df.drop(columns=[case_id_name], errors='ignore')
    
    y1 = df.label
    y2 = df.sigmoid_mm

    if columns_to_remove is not None:
        df = df.drop(columns=columns_to_remove, axis="columns", errors='ignore')

    X = df.drop(columns=["label", "sigmoid_mm", "outcome"], errors='ignore')
    
    return X, y1, y2

def filter_features(features, dataset_columns, feature_type):
    """
    Filters a list of features to keep only those present in the dataset columns.

    It also prints a warning for any features that are missing from the dataset.

    Args:
        features (list of str): The list of feature names to check.
        dataset_columns (list or pandas.Index): The available columns in the dataset.
        feature_type (str): A descriptive string of the feature type (e.g., 'continuous', 'categorical') used for the warning message.

    Returns:
        list of str: A list of feature names that are present in the dataset columns.
    """
    present = [f for f in features if f in dataset_columns]
    missing = [f for f in features if f not in dataset_columns]
    if missing:
        print(f"Warning: the following {feature_type} features are not in training data and will be skipped: {missing}")
    return present

def train_ml_model(train_data, test_data, case_id_name, columns_to_remove,
                   continuous_features, categorical_features, case_study=None, params=None):
    """
    Trains and optimizes machine learning models using CatBoost and Optuna.

    This function processes the data, sets up a scikit-learn ColumnTransformer pipeline for continuous 
    and categorical features, and runs hyperparameter optimization via Optuna for both classification 
    ('label') and regression ('sigmoid_mm') targets. It then trains final models, evaluates their 
    performance, and serializes the resulting pipelines and best parameters to disk.

    Args:
        train_data (pandas.DataFrame): The training dataset.
        test_data (pandas.DataFrame): The test dataset.
        case_id_name (str): The name of the case ID column to drop.ll
        columns_to_remove (list of str): Columns to explicitly remove from the feature set.
        continuous_features (list of str): A list of continuous feature names.
        categorical_features (list of str): A list of categorical feature names.
        case_study (str, optional): The name of the case study, used to define the output directory path. Defaults to None.
        params (dict, optional): A dictionary of configuration parameters including 'optuna_trials', 'early_stopping_rounds', and 'search_spaces'. Defaults to None.

    Returns:
        dict: {"label": {...}, "sigmoid_mm": {...}}, one entry per target with
        the metric name used (Logloss/RMSE), the winning Optuna trial number
        and validation score, the best hyperparameters, the number of trees
        used for the final refit, and the train/test scores of the final
        model -- everything needed to write a training report without having
        to re-parse any file.
    """
    ##########################################
    # SETUP
    ##########################################
    if params is None:
        params = {}

    optuna_trials = params.get("optuna_trials", 80)
    optuna_timeout = params.get("optuna_timeout", 1200)
    early_stopping_rounds = params.get("early_stopping_rounds", 50)
    
    all_search_spaces = params.get("search_spaces", {})
    if all_search_spaces and not any(k in ("label", "sigmoid_mm") for k in all_search_spaces):
        all_search_spaces = {"label": all_search_spaces, "sigmoid_mm": all_search_spaces}

    X_train_raw, y_train1, y_train2 = prepare_df_for_ml(train_data, case_id_name,  columns_to_remove)
    X_test_raw,  y_test1,  y_test2 = prepare_df_for_ml(test_data, case_id_name,  columns_to_remove)

    continuous_features = filter_features(continuous_features, X_train_raw.columns, "continuous")
    categorical_features = filter_features(categorical_features, X_train_raw.columns, "categorical")

    ##########################################
    # PREPROCESSING
    ##########################################
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    print("Pre-processing features...")
    X_train_trans = transformations.fit_transform(X_train_raw)
    X_test_trans = transformations.transform(X_test_raw)

    ##########################################
    # TRAINING
    ##########################################
    results = {}
    for y_train, y_test in [(y_train1, y_test1), (y_train2, y_test2)]:
        print(f"\n--- Optuna Hyperparameter Optimization for: {y_train.name} ---")

        is_regression_target = (y_train.name == "sigmoid_mm")
        search_spaces_config = all_search_spaces.get(y_train.name, {})

        # class balancing 
        if y_train.name == "label":
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            calculated_balance = float(n_neg / n_pos) if n_pos > 0 else 1.0
        else:
            calculated_balance = 1.0

        # catboost parameters
        const_params = {
            "task_type": "CPU",
            "iterations": 3000,
            "early_stopping_rounds": early_stopping_rounds,
            "logging_level": "Silent",                      # not a hyperparameter just a parameter for how to save data
            "allow_writing_files": False,                   # not a hyperparameter just a parameter for how to save data
        }
        
        if y_train.name == "label":
            const_params.update({
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "posterior_sampling": True,
            })
        else:
            const_params.update({
                "loss_function": "RMSEWithUncertainty",
                "eval_metric": "RMSE",
                "posterior_sampling": True,
            })
        
        def objective(trial):
            """
            Objective function for Optuna hyperparameter optimization.
            This function samples hyperparameters from the defined search spaces, configures and 
            trains a CatBoost model (Classifier or Regressor depending on the target), and evaluates 
            its performance on an internal validation set. 

            Args:
                trial (optuna.trial.Trial): An Optuna trial object used to sample hyperparameters.

            Returns:
                float: The eval_metric score (Logloss for classification, RMSE for regression) at the best iteration, which Optuna will attempt to minimize.
            """
            ##########################################
            # DEFINITION OF PARAMETER'S SEARCH
            ##########################################
            trial_params = const_params.copy()

            for param_name, config in search_spaces_config.items():
                p_type = config.get("type")
                if p_type == "float":
                    trial_params[param_name] = trial.suggest_float(param_name, config["min"], config["max"], log=config.get("log", False))
                elif p_type == "int":
                    trial_params[param_name] = trial.suggest_int(param_name, config["min"], config["max"])
                elif p_type == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, config["choices"])

            bootstrap_type = trial_params.get("bootstrap_type")
            if bootstrap_type == "Bayesian" and "bagging_temperature" not in search_spaces_config:
                trial_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
            elif bootstrap_type in ("Bernoulli", "MVS") and "subsample" not in search_spaces_config:
                trial_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            if (trial_params.get("grow_policy") in ("Depthwise", "Lossguide") and "min_data_in_leaf" not in search_spaces_config):
                trial_params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 200, log=True)

            if y_train.name == "label":
                scale_low = max(0.1, calculated_balance * 0.5)
                scale_high = max(scale_low, calculated_balance * 1.5)
                trial_params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", scale_low, scale_high
                )

            ##########################################
            # VALIDATION set
            ##########################################
            X_tr, X_val, y_tr, y_val = extract_internal_running_validation(
                X_trans=X_train_trans, 
                y_train=y_train, 
                train_data=train_data, 
                case_id_name=case_id_name, 
                train_ratio=0.8
            )

            ##########################################
            # TRAINING
            ##########################################
            if y_train.name == "label":
                model = CatBoostClassifier(**trial_params)
            else:
                model = CatBoostRegressor(**trial_params)
            pruning_callback = CatBoostPruningCallback(trial, const_params["eval_metric"]) #pruning
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=0,
                callbacks=[pruning_callback]
            )
            pruning_callback.check_pruned()
            # Save the number of trees early stopping picked for this trial, so the winning trial's tree count can be reused for the final refit on 100% of the training data (see below).
            trial.set_user_attr("best_iteration", int(model.get_best_iteration()))

            ##########################################
            # CALIBRATION
            ##########################################
            if is_regression_target:
                # Applies a post-hoc uncertainty recalibration factor on a validation slice to optimize model uncertainty by minimizing Gaussian NLL. Since the final model is retrained on the full dataset, this serves as an approximation, whereas split-conformal prediction offers rigorous coverage guarantees.
                val_pred = np.asarray(model.predict(X_val), dtype=float)
                resid = y_val.to_numpy(dtype=float) - val_pred[:, 0]
                sigma = np.sqrt(np.clip(val_pred[:, 1], 1e-12, None))
                trial.set_user_attr("sigma_scale", _fit_sigma_scale(resid, sigma))
            else:
                # Temperature-scaling factor for the classifier, fitted on this trial's held-out validation slice:
                # p_cal = sigmoid(raw_logit / T).
                val_logits = np.asarray(model.predict(X_val, prediction_type="RawFormulaVal"), dtype=float) #logits
                trial.set_user_attr("temperature", _fit_temperature(val_logits, y_val.to_numpy(dtype=float)))

            return model.get_best_score()["validation"][const_params["eval_metric"]]

        ##########################################
        # OPTUNA 
        ##########################################
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
            direction="minimize"
        )
        study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout if optuna_timeout else None)
        n_trials_run = len(study.trials)
        n_trials_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if n_trials_complete < optuna_trials:
            print(f"[WARN] only {n_trials_complete}/{optuna_trials} Optuna trials completed for {y_train.name} (timeout hit or trials pruned).")
        print(f"Best trial found for {y_train.name} with score {study.best_value:.5f}")

        ##########################################
        # FINAL TRAINING ON WHOLE DATASET
        ##########################################
        best_iteration = study.best_trial.user_attrs["best_iteration"]
        final_params = const_params.copy()
        final_params.update(study.best_params)
        final_params["iterations"] = best_iteration + 1
        final_params.pop("early_stopping_rounds", None)
        final_params["logging_level"] = "Verbose"

        if y_train.name == "label":
            final_model = CatBoostClassifier(**final_params)
        else:
            final_model = CatBoostRegressor(**final_params)
        final_model.fit(X_train_trans, y_train, verbose=500)

        # We reuse the sigma and temperature fitted on the validation set of the best iteration
        if is_regression_target:
            sigma_scale = float(study.best_trial.user_attrs.get("sigma_scale", 1.0))
            prediction_step = UncertaintyRegressor(final_model, sigma_scale=sigma_scale)
        else:
            temperature = float(study.best_trial.user_attrs.get("temperature", 1.0))
            prediction_step = UncertaintyClassifier(final_model, temperature=temperature)
        print("\n[INFO] Training complete. Evaluating performance...")

        # uncertainty of the last fit
        uncertainty_report = None
        if y_train.name == "label":
            metric_name = "Logloss"
            y_train_proba = prediction_step.predict_proba(X_train_trans)[:, 1]
            y_test_proba = prediction_step.predict_proba(X_test_trans)[:, 1]
            train_score = log_loss(y_train, y_train_proba)
            test_score = log_loss(y_test, y_test_proba)
            print("Logloss score of training set:", train_score)
            print("Logloss score of test set:", test_score)

            # Classifier uncertainty (entropy decomposition) + temperature calibration, on the test set. 
            y_test_arr = y_test.to_numpy(dtype=float)
            cu = prediction_step.predict_uncertainty(X_test_trans)
            total_var = float(np.mean(cu["total_entropy"]))
            uncertainty_report = {
                "temperature": temperature,
                "logloss_raw": test_score,
                "logloss_calibrated": float(log_loss(y_test, cu["proba_calibrated"])),
                "ece_raw": _expected_calibration_error(y_test_arr, cu["proba"]),
                "ece_calibrated": _expected_calibration_error(y_test_arr, cu["proba_calibrated"]),
                "mean_data_entropy": float(np.mean(cu["data_entropy"])),
                "mean_knowledge_entropy": float(np.mean(cu["knowledge_entropy"])),
                "mean_total_entropy": total_var,
                "epistemic_entropy_fraction": (
                    float(np.mean(cu["knowledge_entropy"]) / total_var) if total_var > 0 else 0.0
                ),
            }
            print(
                f"Uncertainty (test avg): data/aleatoric entropy = {uncertainty_report['mean_data_entropy']:.5f}, "
                f"knowledge/epistemic = {uncertainty_report['mean_knowledge_entropy']:.5f}, "
                f"total = {uncertainty_report['mean_total_entropy']:.5f} "
                f"(epistemic share {uncertainty_report['epistemic_entropy_fraction'] * 100:.1f}%)"
            )
            print(
                f"Calibration: temperature T = {temperature:.3f} | "
                f"Logloss {test_score:.5f} -> {uncertainty_report['logloss_calibrated']:.5f}, "
                f"ECE {uncertainty_report['ece_raw']:.4f} -> {uncertainty_report['ece_calibrated']:.4f}"
            )
        else:
            metric_name = "RMSE"
            y_train_pred = prediction_step.predict(X_train_trans)
            y_test_pred = prediction_step.predict(X_test_trans)
            train_score = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            test_score = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            train_mae = float(mean_absolute_error(y_train, y_train_pred))
            test_mae = float(mean_absolute_error(y_test, y_test_pred))
            print(f"RMSE / MAE of training set: {train_score:.5f} / {train_mae:.5f}")
            print(f"RMSE / MAE of test set:     {test_score:.5f} / {test_mae:.5f}")

            # Uncertainty diagnostics on the test set
            unc = prediction_step.predict_uncertainty(X_test_trans)
            resid = y_test.to_numpy(dtype=float) - unc["mean"]
            abs_err = np.abs(resid)
            std_cal = unc["total_std"]
            std_raw = std_cal / sigma_scale
            mean_total_var = float(np.mean(std_cal ** 2))
            
            calib_cal = _regression_calibration_metrics(resid, std_cal)
            calib_raw = _regression_calibration_metrics(resid, std_raw)
            uncertainty_report = {
                "sigma_scale": sigma_scale,
                "mean_data_std": float(np.mean(unc["data_std"])),
                "mean_knowledge_std": float(np.mean(unc["knowledge_std"])),
                "mean_std": float(np.mean(std_cal)),
                "epistemic_var_fraction": (
                    float(np.mean(unc["knowledge_std"] ** 2) / mean_total_var) if mean_total_var > 0 else 0.0
                ),
                "ence_raw": calib_raw["ence"],
                "ence": calib_cal["ence"],
                "cv": calib_cal["cv"],
                "calib_n_bins": calib_cal["n_bins"],
                "rmv_bins": calib_cal["rmv_bins"],
                "rmse_bins": calib_cal["rmse_bins"],
                "rmv_bins_raw": calib_raw["rmv_bins"],
                # coverage with CatBoost's raw sigma vs the recalibrated sigma
                "coverage_1sigma_raw": float(np.mean(abs_err <= std_raw)),
                "coverage_2sigma_raw": float(np.mean(abs_err <= 2.0 * std_raw)),
                "coverage_1sigma": float(np.mean(abs_err <= std_cal)),
                "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_cal)),
                "test_mae": test_mae,
            }
            print(
                f"Uncertainty (test avg): data/aleatoric std = {uncertainty_report['mean_data_std']:.5f}, "
                f"knowledge/epistemic std = {uncertainty_report['mean_knowledge_std']:.5f}, "
                f"total std = {uncertainty_report['mean_std']:.5f} "
                f"(epistemic share {uncertainty_report['epistemic_var_fraction'] * 100:.1f}%) | "
                f"median (sharpness) = {uncertainty_report['median_std']:.5f}"
            )
            print(
                f"Calibration: sigma_scale = {sigma_scale:.3f} | "
                f"ENCE {uncertainty_report['ence_raw'] * 100:.2f}% -> {uncertainty_report['ence'] * 100:.2f}% | "
                f"c_v {uncertainty_report['cv']:.3f} | "
                f"cov +/-1s {uncertainty_report['coverage_1sigma_raw']:.3f} -> {uncertainty_report['coverage_1sigma']:.3f}, "
                f"+/-2s {uncertainty_report['coverage_2sigma_raw']:.3f} -> {uncertainty_report['coverage_2sigma']:.3f}"
            )
        print("--------------------------------------------------")

        best_pipeline = Pipeline(steps=[
            ("transformation", transformations),
            ("prediction", prediction_step)
        ])

        output_dir = Path(f"./case_studies/{case_study}/model")
        joblib.dump(best_pipeline, output_dir / f"catboost_model_{y_train.name}.joblib")

        with open(output_dir / f"best_hyperparams_{y_train.name}.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)

        results[y_train.name] = {
            "metric_name": metric_name,
            "n_trials_run": n_trials_run,
            "n_trials_complete": n_trials_complete,
            "best_trial_number": study.best_trial.number,
            "best_validation_score": study.best_value,
            "best_params": study.best_params,
            "n_trees_final_model": final_params["iterations"],
            "train_score": float(train_score),
            "test_score": float(test_score),
            "uncertainty": uncertainty_report,
        }

    return results