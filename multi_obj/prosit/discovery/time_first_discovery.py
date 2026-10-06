"""
Discovery of the "time first, then route" models used by SimulatorEngine.

For every place of the net the simulator learns two things from the aligned log:

- a TIME model: how long after the case arrives in the place (end of its last visible
  event) its next visible event starts, whatever that event is. It is a regression tree
  on log(1 + days); simulation draws one of the waiting times observed in the leaf of the
  case, which makes it easy to condition on a lower bound (a test case already idle for
  some days at the train/test split cannot have its next event earlier than that);
- for decision places (more than one outgoing transition), a ROUTE model: which outgoing
  transition is taken, given the case history, its attributes, its age and the waiting
  time drawn by the time model. It is a single multiclass tree per place, so the branches
  compete with each other (one binary model per transition, as in cf_discovery.py, cannot
  shift probability from one branch to another).

Drawing the time before the branch mirrors how these processes evolve: a case waits, and
what happens next depends on how long it has waited (e.g. an application with no answer
from the customer after 30 days is cancelled).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

AGE_FEATURE = "CASE_AGE_DAYS"    # days from the case start to its arrival in the place
WAIT_FEATURE = "WAIT_DAYS"       # days from the arrival in the place to the next visible event

MIN_ROWS_PER_PLACE = 20
TIME_TREE_PARAMS = {"max_depth": 5, "min_samples_leaf": 50}
ROUTE_TREE_GRID = {"max_depth": [2, 3, 4, 6, 8], "min_samples_leaf": [10, 30, 100]}


def feature_columns(label_data_attributes, label_data_attributes_categorical, attribute_values_label_categorical,
                    net_transition_labels):
    """Ordered feature names shared by the time and route models (route models append WAIT_FEATURE).

    Categorical attributes are one-hot encoded with the same "<attribute> = <value>" names the
    simulator uses for case attributes; activity labels are sorted so the order does not depend
    on set iteration order (net_transition_labels is built from a set).
    """
    cols = []
    for a in label_data_attributes:
        if a in label_data_attributes_categorical:
            cols += [f"{a} = {v}" for v in attribute_values_label_categorical[a]]
        else:
            cols.append(a)
    return cols + sorted(net_transition_labels) + [AGE_FEATURE]


def add_time_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """Adds AGE_FEATURE, WAIT_FEATURE and "fired" (name of the fired transition) to the rows
    built by build_df_features, which are in firing order within each case."""
    df = df_features.copy()
    start = pd.to_datetime(df["start_t"], utc=True, errors="coerce")
    end = pd.to_datetime(df["end_t"], utc=True, errors="coerce")
    case_start = start.groupby(df["case_id"]).transform("min")
    # arrival in the place = end of the last visible event before this row (case start for the first one)
    arrival = end.groupby(df["case_id"]).transform(lambda s: s.shift().ffill()).fillna(case_start)
    # next visible event = this row's event if visible, otherwise the next visible one of the case;
    # none (the case ends through silent transitions) -> the case closes right away, wait 0
    next_visible_start = start.where(df["transition_label"].notna()).groupby(df["case_id"]).bfill()
    df[AGE_FEATURE] = ((arrival - case_start).dt.total_seconds() / 86400).fillna(0.0)
    df[WAIT_FEATURE] = ((next_visible_start - arrival).dt.total_seconds() / 86400).fillna(0.0).clip(lower=0.0)
    df["fired"] = df["transition"].apply(lambda t: t.name)
    return df


def discover_time_first_models(df_features: pd.DataFrame, net, label_data_attributes, label_data_attributes_categorical,
                               attribute_values_label_categorical, net_transition_labels) -> dict:
    """Fits the time model of every place and the route model of every decision place.

    Returns:
        dict with keys "features" (ordered feature names), "time" {place name: {"tree", "leaf",
        "all"}} and "route" {place name: {"tree"} or {"constant": transition name}}.
    """
    features = feature_columns(label_data_attributes, label_data_attributes_categorical,
                               attribute_values_label_categorical, net_transition_labels)
    df = add_time_features(df_features)
    for a in label_data_attributes_categorical:
        for v in attribute_values_label_categorical[a]:
            df[f"{a} = {v}"] = (df[a] == v).astype(int)

    time_models, route_models = {}, {}
    for place in net.places:
        outs = {a.target.name for a in place.out_arcs}
        rows = df[df["fired"].isin(outs)]
        if len(rows) < MIN_ROWS_PER_PLACE:
            continue
        X = rows[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        wait = rows[WAIT_FEATURE].values
        reg = DecisionTreeRegressor(random_state=72, **TIME_TREE_PARAMS).fit(X, np.log1p(wait))
        leaves = reg.apply(X)
        time_models[place.name] = {"tree": reg, "all": wait,
                                   "leaf": {leaf: wait[leaves == leaf] for leaf in np.unique(leaves)}}
        if len(place.out_arcs) < 2:
            continue
        if rows["fired"].nunique() < 2:
            route_models[place.name] = {"constant": rows["fired"].iloc[0]}
            continue
        Xr = np.column_stack([X, wait])
        try:
            tree = GridSearchCV(DecisionTreeClassifier(random_state=72), ROUTE_TREE_GRID, cv=3,
                                scoring="neg_log_loss").fit(Xr, rows["fired"]).best_estimator_
        except ValueError:
            tree = DecisionTreeClassifier(random_state=72, max_depth=4, min_samples_leaf=30).fit(Xr, rows["fired"])
        route_models[place.name] = {"tree": tree}
    return {"features": features, "time": time_models, "route": route_models}


def feature_vector(models: dict, case_attributes: dict, case_history: dict, age_days: float) -> list:
    """Case features in the order the models were trained with (WAIT_FEATURE excluded)."""
    values = dict(case_attributes) | dict(case_history) | {AGE_FEATURE: age_days}
    out = []
    for f in models["features"]:
        try:
            out.append(float(values.get(f, 0.0) or 0.0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def sample_wait_days(models: dict, place_name: str, x: list, min_days: float, rng) -> float:
    """Draws the waiting time (days) of a case in a place from the waiting times observed in the
    leaf of its time tree, keeping only those >= min_days; falls back to the whole place, then to
    min_days itself."""
    tm = models["time"][place_name]
    values = tm["leaf"].get(tm["tree"].apply(np.array([x]))[0], tm["all"])
    candidates = values[values >= min_days]
    if len(candidates) == 0:
        candidates = tm["all"][tm["all"] >= min_days]
    return float(rng.choice(list(candidates))) if len(candidates) else float(min_days)


def route_probabilities(models: dict, place_name: str, x: list, wait_days: float) -> dict:
    """{transition name: probability} of the branches of a decision place."""
    rm = models["route"][place_name]
    if "constant" in rm:
        return {rm["constant"]: 1.0}
    proba = rm["tree"].predict_proba(np.array([list(x) + [wait_days]]))[0]
    return dict(zip(rm["tree"].classes_, proba))
