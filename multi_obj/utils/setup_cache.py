"""On-disk cache for the expensive, run-invariant part of the recommendation setup.

`transition_system()` iterates the training log row by row and takes tens of
seconds to a couple of minutes per case study. Its result depends only on the
training data and the window size -- neither changes between repeated runs of
3_plot_pareto / 3_run_experiment / 3_tune_nsga2_params -- so it is cached here
and reloaded instead of recomputed.
"""

import time
from pathlib import Path

import joblib

from utils.transition_system import transition_system

# Bump when transition_system()'s logic changes, to invalidate every cache file.
CACHE_VERSION = 1

CASE_ID_NAME = "case:concept:name"
ACTIVITY_COLUMN_NAME = "concept:name"


def get_transition_graph(
    case_study,
    train_data,
    *,
    case_id_name=CASE_ID_NAME,
    activity_column_name=ACTIVITY_COLUMN_NAME,
    window_size=5,
    base_dir=".",
    rebuild=False,
):
    """Return transition_system()'s graph for (case_study, window_size), cached on disk.

    The cache lives at
    ``case_studies/<case_study>/cache/transition_ws{window_size}.joblib`` and is
    reused as long as it is newer than ``train_data.csv`` and CACHE_VERSION is
    unchanged. Pass ``rebuild=True`` to force recomputation (e.g. after editing
    transition_system()).

    Only the graph is returned; transition_system()'s second output (the
    next-activity frequency dict) is not used anywhere in the codebase.
    """
    case_dir = Path(base_dir) / "case_studies" / case_study
    cache_dir = case_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"transition_ws{window_size}.joblib"
    train_csv = case_dir / "train_data.csv"

    if not rebuild and cache_path.exists():
        cache_is_fresh = (not train_csv.exists()) or train_csv.stat().st_mtime <= cache_path.stat().st_mtime
        if cache_is_fresh:
            try:
                bundle = joblib.load(cache_path)
                if bundle.get("version") == CACHE_VERSION and bundle.get("window_size") == window_size:
                    print(f"[setup_cache] transition graph loaded from cache/{cache_path.name}")
                    return bundle["transition_graph"]
            except Exception as exc:  # noqa: BLE001 - a broken cache must not be fatal
                print(f"[setup_cache] cache unreadable ({exc}); rebuilding")

    print(f"[setup_cache] building transition system for {case_study} (window_size={window_size})...")
    t0 = time.time()
    transition_graph, _ = transition_system(
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size,
    )
    joblib.dump(
        {"version": CACHE_VERSION, "window_size": window_size, "transition_graph": transition_graph},
        cache_path,
        compress=3,
    )
    print(f"[setup_cache] built in {time.time() - t0:.1f}s, cached to cache/{cache_path.name}")
    return transition_graph
