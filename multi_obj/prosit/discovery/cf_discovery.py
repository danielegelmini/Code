from tqdm import tqdm
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from prosit.utils.rule_utils import DecisionRules


# Applied only to the routing/control-flow decision trees below (not the execution/waiting-time
# regressors in time_discovery.py, which already fit a distribution per leaf from the leaf's own
# samples rather than a single point probability). A leaf right at the min_samples_leaf floor (100,
# see build_models) is shrunk about halfway toward the transition's global rate; a leaf with far
# more supporting samples stays close to its own raw fraction. See _shrink_leaf_probabilities.
LEAF_SHRINKAGE_PSEUDOCOUNT = 100


def discover_weight_transitions(
        df_features: pd.DataFrame,
        net_transition_labels: list, 
        max_depths_cv: list = range(1, 6),
        label_data_attributes: list = [], 
        label_data_attributes_categorical: list = [], 
        values_categorical: dict = dict(),
        transition_model_type: str = 'DecisionTree'
    ) -> dict :

    if not max_depths_cv:
        transitions = df_features['transition'].unique()
        transition_weights = {t: (df_features["transition"] == t).sum() / df_features["prev_enabled_transitions"].apply(lambda t_set: t in t_set).sum() for t in transitions}
    else:
        transition_weights = build_models(
                                            df_features,
                                            net_transition_labels,
                                            label_data_attributes,
                                            label_data_attributes_categorical,
                                            values_categorical,
                                            model_type=transition_model_type,
                                            max_depths_cv=max_depths_cv
                                        )

    return transition_weights


def _shrink_leaf_probabilities(
        clf: DecisionRules,
        decision_tree: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        pseudo_count: float = LEAF_SHRINKAGE_PSEUDOCOUNT,
    ) -> None:
    """Empirical-Bayes ("credibility") shrinkage of each leaf's firing probability toward the
    transition's own global (unconditioned) rate, weighted by how much data actually supports
    that leaf:

        p_leaf = (successes_in_leaf + pseudo_count * global_rate) / (n_in_leaf + pseudo_count)

    Why: clf.rules[leaf]['value'] (set by DecisionRules.from_decision_tree, via a graphviz-export
    round trip) is the RAW, unregularized class-1 fraction observed in that leaf -- fine when the
    leaf has plenty of samples, but for a genuinely RARE transition (a small global_rate, e.g. a
    rework-loop re-entry point taken in only a few percent of opportunities) that raw fraction is
    estimated from a small, feature-partitioned subsample of an already-rare event and can overshoot
    the true rate substantially -- classic small-sample bias for rare-event estimation. Observed on
    BPI12: a control-flow gate's raw leaf fraction was 3-6x the global rate for the common branch,
    which (since the gate is revisited many times within a single simulated case) reliably pushed
    the rework loop it guards into firing repeatedly within a case, though it is taken at most once
    per case in the real log. This does not touch the tree's learned STRUCTURE (which feature/
    threshold to split on stays whatever the fit + min_samples_leaf/min_samples_split already
    regularize) -- only the per-leaf point estimate, so history-conditioned differentiation between
    leaves is preserved, just calibrated toward the transition's own base rate.

    Mutates clf.rules in place; generic across every transition and every case study (no BPI12- or
    activity-specific logic).
    """
    if not isinstance(clf.rules, dict):
        return  # single-value / non-tree clf -- nothing to shrink
    global_rate = float(y.mean())
    leaf_ids = decision_tree.apply(X)
    y_arr = y.to_numpy()
    for leaf_id in set(leaf_ids):
        node = clf.rules.get(int(leaf_id))
        if node is None or 'value' not in node:
            continue  # not a leaf in the parsed rules (shouldn't happen, but stay defensive)
        mask = leaf_ids == leaf_id
        n = int(mask.sum())
        successes = float(y_arr[mask].sum())
        node['value'] = (successes + pseudo_count * global_rate) / (n + pseudo_count)


def build_models(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        model_type: str = 'DecisionTree', 
        max_depths_cv: list = range(1,6)
    ) -> dict :
    
    datasets_t = build_training_datasets(
                    df_features,
                    net_transition_labels, 
                    label_data_attributes
                )

    param_grid = {'max_depth': max_depths_cv}

    models_t = dict()
    
    for t in tqdm(datasets_t.keys()):
        data_t = datasets_t[t]
        if len(data_t['class'].unique())<2:
            # Degenerate case: whenever this transition was enabled, it EITHER always fired
            # (class always 1) or never did (class always 0) -- no variance to fit a tree on.
            # Must preserve which of the two, not blank it to None: None is later defaulted to a
            # constant weight of 1 regardless (decision_rules_to_dict's `if d is None: d = 1`),
            # silently turning a "this never actually fires here" transition into "always fires
            # with full weight" -- observed on BPI12's A_CANCELLED, whose one enabling point
            # (a rework-loop exit gate) has class constantly 0 in the aligned log (matching
            # discover_weight_transitions' own depth-0/flat formula, which gives it weight 0 --
            # see the `if not max_depths_cv` branch above), yet was being forced to weight 1 here,
            # making a transition the real log essentially never takes at that gate dominate the
            # gate's competing weighted draw and fire repeatedly within a single simulated case.
            models_t[t] = float(data_t['class'].mode().iloc[0])
            continue
        
        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_t[a+' = '+str(v)] = (data_t[a] == v).astype(int)
            del data_t[a]

        X = data_t.drop(columns=['class'])
        y = data_t['class']

        if model_type == 'LogisticRegression':
            clf_t = LogisticRegression(random_state=72).fit(X, y)

        elif model_type == 'DecisionTree':

            # min_samples_leaf/min_samples_split match the regularization already used for the
            # execution/waiting-time regression trees (time_discovery.py's build_models_ex/wt).
            # Without it, an unconstrained classifier at these shallow depths (1-2) reliably picks
            # a spurious split on one of the ~hundreds of sparse one-hot label_data_attributes
            # dummy columns (e.g. a single rare 'AMOUNT_REQ = <value>') over the handful of
            # genuinely informative case-history columns, just because a tiny, low-impurity leaf
            # is easy to carve out of a few samples by chance -- this was directly observed on
            # BPI12: a routing gate split on 'AMOUNT_REQ = 21000' instead of loop-repeat history,
            # leaving the common branch's probability essentially unregularized noise instead of a
            # real history-conditioned estimate, and driving simulated rework loops far longer than
            # in the real log.
            if max_depths_cv:
                clf_t_dtc = DecisionTreeClassifier(random_state=72, min_samples_leaf=100, min_samples_split=200)
                try:
                    grid_search = GridSearchCV(estimator=clf_t_dtc, param_grid=param_grid, cv=3).fit(X, y)
                    clf_t_dtc = grid_search.best_estimator_
                except:
                    clf_t_dtc = DecisionTreeClassifier(max_depth=2, random_state=72, min_samples_leaf=100, min_samples_split=200)
                    clf_t_dtc.fit(X, y)
            else:
                clf_t_dtc = DecisionTreeClassifier(random_state=72, max_depth=1, min_samples_leaf=100, min_samples_split=200)
                clf_t_dtc.fit(X, y)

            clf_t = DecisionRules()
            clf_t.from_decision_tree(clf_t_dtc)
            _shrink_leaf_probabilities(clf_t, clf_t_dtc, X, y)

        if clf_t is None:
            clf_t = float(y.mode().iloc[0])

        models_t[t] = clf_t
    
    return models_t



def build_training_datasets(
        df_features: pd.DataFrame,
        net_transition_labels: list, 
        label_data_attributes: list
    ) -> dict:

    df_cf = df_features[["transition"] + ["prev_enabled_transitions"] + label_data_attributes + net_transition_labels]

    df_cf = df_cf.explode('prev_enabled_transitions')
    df_cf['class'] = (df_cf['prev_enabled_transitions'] == df_cf['transition']).astype(int)

    df_cf = df_cf.drop(columns=['transition'])
    df_cf = df_cf.rename(columns={'prev_enabled_transitions': 'transition'})
        
    net_transitions = df_cf['transition'].unique()
    datasets_t = {t: df_cf[df_cf["transition"] == t].drop(columns=['transition']).reset_index(drop=True) for t in net_transitions}

    return datasets_t
