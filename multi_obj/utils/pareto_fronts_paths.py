"""Where the figures of the Pareto fronts are saved, and the clean-up of the old ones.

Every figure of a Pareto front lives under multi_obj/pareto_fronts/, one folder per kind:

    pareto_fronts/
        pareto_fronts_dataset_predicted/<case_study>/       3_plot_pareto.py (the front the recommender builds)
        pareto_fronts_predicted_vs_simulated/<case_study>/  8_plot_case_comparison.py
        pareto_fronts_bpi12_vs_bpi12sim/                     8e_plot_real_vs_sim_approach.py

Before writing its figures, each script deletes (by default) the figures of the same kind already in its folder,
only the ones it would itself produce (its own filename pattern, and only for the case study it is run on), so the
folder always holds the output of the last run and nothing else.
"""
from pathlib import Path

PARETO_FRONTS_DIR = Path(__file__).resolve().parent.parent / "pareto_fronts"
DATASET_PREDICTED_DIR = PARETO_FRONTS_DIR / "pareto_fronts_dataset_predicted"
PREDICTED_VS_SIMULATED_DIR = PARETO_FRONTS_DIR / "pareto_fronts_predicted_vs_simulated"
BPI12_VS_BPI12SIM_DIR = PARETO_FRONTS_DIR / "pareto_fronts_bpi12_vs_bpi12sim"


def clear_old_figures(folder: Path, pattern: str) -> int:
    """Delete the files of `folder` matching the glob `pattern` (not recursive); returns how many were deleted."""
    folder = Path(folder)
    if not folder.exists():
        return 0
    old = [f for f in folder.glob(pattern) if f.is_file()]
    for f in old:
        f.unlink()
    if old:
        print(f"Deleted {len(old)} old figure(s) {pattern} in {folder}")
    return len(old)
