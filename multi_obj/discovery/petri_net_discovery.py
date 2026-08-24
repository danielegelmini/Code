#!/usr/bin/env python3
"""
petri_net_discovery.py
=======================

Standalone script that automatically discovers a Petri net for every event log
found under the `case_studies/<dataset>/log_<dataset>.xes` folders (same
layout used in the original notebook, but using the full XES log instead of
just the training split), using the Split Miner algorithm. Optuna (Bayesian
optimization) is used to find the (epsilon, eta) combination that maximizes
the F-score (harmonic mean of alignment-based fitness and precision).

For every log found it saves, in the same case study's `discovery_output/`
directory (overwriting any previous Petri net there, since that is where
4_run_recommendation_simulation.py loads it from):
    - <dataset>_best_petri_net.pnml   -> the discovered Petri net (PNML format)
    - <dataset>_best_petri_net.jpg    -> a rendered image of the Petri net

Usage
-----
    python petri_net_discovery.py
        -> automatically finds and processes every log_<dataset>.xes under
           <base_dir>/case_studies/<dataset>/

    python petri_net_discovery.py --n-trials 50
        -> same, but with a custom number of Optuna trials per log

    python petri_net_discovery.py log1.xes log2.xes ...
        -> (optional) explicitly process only the given XES files instead of
           auto-discovering them
"""

import argparse
import io
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pm4py
import tqdm
from pm4py.algo.discovery.split_miner import algorithm as split_miner
from pm4py.algo.discovery.split_miner.variants import classic as split_miner_classic
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# ---------------------------------------------------------------------------
# Disable ALL tqdm progress bars globally.
#
# pm4py internally uses tqdm to show a progress bar while computing
# alignment-based fitness/precision (pm4py.fitness_alignments /
# pm4py.precision_alignments). Since these are called once per Optuna trial,
# with 30+ trials per log you'd otherwise get dozens of progress bars
# cluttering the output. Passing "show_progress_bar": False through pm4py's
# `parameters` dict is version-dependent and easy to get wrong (same pitfall
# as the epsilon/eta enum keys), so instead we monkeypatch tqdm itself to
# force disable=True on every progress bar it creates, regardless of who
# creates it or which parameter name they used.
_original_tqdm_init = tqdm.tqdm.__init__


def _silent_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _original_tqdm_init(self, *args, **kwargs)


tqdm.tqdm.__init__ = _silent_tqdm_init

# ---------------------------------------------------------------------------
# Case study auto-discovery (mirrors the logic used in the original notebook)
# ---------------------------------------------------------------------------

# Name pattern of the XES file expected inside each case-study folder, e.g.
# "case_studies/BAC/log_BAC.xes".
LOG_FILE_NAME_TEMPLATE = "log_{dataset}.xes"

# Fallback project path used during development, in case the "case_studies"
# folder is not found relative to the current working directory.
FALLBACK_BASE_DIR = Path("C:/Users/Utente/Desktop/tesi magistrale/Code/2-gelmini-multi")


def resolve_base_dir() -> Path:
    """Resolve the project base directory, same logic as the notebook."""
    base_dir = Path.cwd()
    if not (base_dir / "case_studies").exists():
        base_dir = FALLBACK_BASE_DIR.resolve()
    return base_dir


def discover_case_study_logs(base_dir: Path):
    """Find every `log_<dataset>.xes` under `<base_dir>/case_studies/<dataset>/`.

    Returns a list of (dataset_name, xes_path) tuples, sorted by dataset name.
    """
    case_studies_dir = base_dir / "case_studies"
    if not case_studies_dir.exists():
        return []

    logs = []
    for dataset_dir in sorted(case_studies_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        xes_path = dataset_dir / LOG_FILE_NAME_TEMPLATE.format(dataset=dataset_dir.name)
        if xes_path.exists():
            logs.append((dataset_dir.name, xes_path))

    return logs


# ---------------------------------------------------------------------------
# Event log loading
# ---------------------------------------------------------------------------


def load_event_log(xes_path: str):
    """Read a PM4Py-compatible XES log and convert it into a PM4Py EventLog object."""
    return pm4py.read_xes(str(xes_path))


# ---------------------------------------------------------------------------
# Petri net rendering helper
# ---------------------------------------------------------------------------


def _clean_graphviz_body(gviz):
    """Clean up the graphviz labels/attributes of visible (non-silent) transitions."""
    new_body = []
    for line in gviz.body:
        is_silent = (
            "fillcolor=black" in line
            or 'fillcolor="#000000"' in line
            or 'label=""' in line
        )
        is_box = "shape=box" in line or 'shape="box"' in line

        if is_box and not is_silent:
            match_label = re.search(r'label="(.*?)"', line)
            if match_label:
                raw_text = match_label.group(1)
                clean_text = (
                    raw_text.replace("\\n", " ").replace("\\l", " ").replace("\\r", " ").strip()
                )
                line = re.sub(r'label=".*?"', f'label="{clean_text}"', line)

            line = re.sub(r'width\s*=\s*"?[\d\.]+"?', "", line)
            line = re.sub(r'height\s*=\s*"?[\d\.]+"?', "", line)
            line = re.sub(r'fixedsize\s*=\s*"?\w+"?', "", line)

            line = re.sub(r",\s*,", ",", line)
            line = line.replace("[,", "[").replace(",]", "]")

            override_attrs = (
                'shape=box, fixedsize=false, fontname="Helvetica", fontsize=13, '
                'margin="0.15,0.08"'
            )
            line = line.replace("shape=box", override_attrs).replace('shape="box"', override_attrs)

        new_body.append(line)

    gviz.body = new_body
    return gviz


def save_petri_net_image(net, initial_marking, final_marking, jpg_path):
    """Render a Petri net to a JPG image at the given path."""
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    gviz = _clean_graphviz_body(gviz)

    png_bytes = gviz.pipe(format="png", quiet=True)

    fig, ax = plt.subplots(figsize=(24, 8))
    img = plt.imread(io.BytesIO(png_bytes))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(jpg_path), format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Split Miner discovery + evaluation
# ---------------------------------------------------------------------------


def discover_petri_net(log, epsilon: float, eta: float):
    """Run Split Miner (classic variant) with the given hyperparameters and
    return (net, im, fm).

    IMPORTANT: pm4py's split_miner.algorithm.apply() does NOT accept
    `epsilon`/`eta` as keyword arguments, and its `parameters` dict does NOT
    use the plain strings "epsilon"/"eta" as keys — it expects the specific
    enum members `classic.Parameters.EPSILON` / `classic.Parameters.ETA`
    (whose actual string values are "split_miner_epsilon" /
    "split_miner_eta"). Passing the wrong keys is silently ignored by pm4py
    and the algorithm falls back to its internal defaults — which is why an
    earlier version of this script always produced identical fitness/
    precision regardless of the epsilon/eta values being tried.
    """
    parameters = {
        split_miner_classic.Parameters.EPSILON: epsilon,
        split_miner_classic.Parameters.ETA: eta,
    }
    bpmn_model = split_miner.apply(
        log, parameters=parameters, variant=split_miner.Variants.CLASSIC
    )

    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_model)
    return net, initial_marking, final_marking


def evaluate_petri_net(log, net, initial_marking, final_marking):
    """Compute (fitness, precision) using alignment-based metrics."""
    fitness_res = pm4py.fitness_alignments(log, net, initial_marking, final_marking)
    fitness = fitness_res.get("log_fitness", fitness_res.get("averageFitness", 0.0))
    precision = pm4py.precision_alignments(log, net, initial_marking, final_marking)
    return fitness, precision


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------


def optimize_hyperparameters(log, n_trials: int, dataset_name: str):
    """Search for the (epsilon, eta) pair that maximizes the F-score."""

    def objective(trial):
        eps = trial.suggest_float("epsilon", 0.0, 1.0, step=0.1)
        eta = trial.suggest_float("eta", 0.0, 1.0, step=0.1)

        try:
            net, im, fm = discover_petri_net(log, eps, eta)
            fitness, precision = evaluate_petri_net(log, net, im, fm)
            print(f"  [Trial] epsilon={eps:.1f}, eta={eta:.1f} | Fitness={fitness:.4f}, Precision={precision:.4f}, F-score={2 * (fitness * precision) / (fitness + precision) if fitness + precision > 0 else 0.0:.4f}")
        except Exception as exc:
            print(f"  [!] Trial with epsilon={eps:.1f}, eta={eta:.1f} failed "
                  f"({type(exc).__name__}: {exc}). Scoring as 0.0.")
            trial.set_user_attr("fitness", 0.0)
            trial.set_user_attr("precision", 0.0)
            trial.set_user_attr("failed", True)
            return 0.0

        trial.set_user_attr("fitness", fitness)
        trial.set_user_attr("precision", precision)

        if fitness + precision == 0:
            return 0.0
        return 2 * (fitness * precision) / (fitness + precision)

    print(f"  [...] Running Optuna hyperparameter search for '{dataset_name}' "
          f"({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_eps = study.best_params["epsilon"]
    best_eta = study.best_params["eta"]
    best_fitness = study.best_trial.user_attrs["fitness"]
    best_precision = study.best_trial.user_attrs["precision"]
    best_f_score = study.best_value

    print(f"  [✓] Best hyperparameters: epsilon={best_eps:.1f}, eta={best_eta:.1f}")
    print(f"  [✓] Fitness={best_fitness:.4f}  Precision={best_precision:.4f}  "
          f"F-score={best_f_score:.4f}")

    return best_eps, best_eta


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_log_file(xes_path: str, n_trials: int, dataset_name: str = None):
    xes_path = Path(xes_path)
    # For auto-discovered logs, use the parent folder name (e.g. "BAC") so
    # that files aren't all named "log_<dataset>_best_petri_net.*". For
    # explicitly-passed files, fall back to the file stem.
    if dataset_name is None:
        dataset_name = xes_path.stem
    output_dir = xes_path.parent  # save results next to the input file

    print(f"=== Processing '{dataset_name}' ({xes_path}) ===")
    log = load_event_log(str(xes_path))

    best_eps, best_eta = optimize_hyperparameters(log, n_trials, dataset_name)

    print("  [...] Generating final Petri net with the best hyperparameters...")
    net, initial_marking, final_marking = discover_petri_net(log, best_eps, best_eta)

    # Save into "discovery_output" -- the directory 4_run_recommendation_simulation.py
    # actually loads the Petri net (and its cached simulator params) from.
    discovery_output_dir = output_dir / "discovery_output"
    discovery_output_dir.mkdir(parents=True, exist_ok=True)
    pnml_path = discovery_output_dir / f"{dataset_name}_best_petri_net.pnml"
    jpg_path = discovery_output_dir / f"{dataset_name}_best_petri_net.jpg"

    pm4py.write_pnml(net, initial_marking, final_marking, str(pnml_path))
    save_petri_net_image(net, initial_marking, final_marking, jpg_path)

    print(f"  [✓] Saved: {pnml_path}")
    print(f"  [✓] Saved: {jpg_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Discover a Petri net (Split Miner) for each event log, "
                    "using Optuna to optimize epsilon/eta hyperparameters. "
                    "By default, every 'log_<dataset>.xes' found under "
                    "case_studies/<dataset>/ is processed automatically."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="(Optional) explicit XES log file(s) to process instead of "
             "auto-discovering the case_studies/<dataset>/log_<dataset>.xes files.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials per log (default: 30).",
    )
    args = parser.parse_args()

    os.environ.pop("GV_FILE_PATH", None)
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.logs:
        # Explicit files passed on the command line: (dataset_name=None, path)
        jobs = [(None, Path(p)) for p in args.logs]
    else:
        # Default behaviour: auto-discover every case study, like the notebook.
        base_dir = resolve_base_dir()
        found = discover_case_study_logs(base_dir)

        if not found:
            print(f"No 'log_<dataset>.xes' files found under "
                  f"'{base_dir / 'case_studies'}'.")
            print("Pass explicit XES paths as arguments instead, e.g.:\n"
                  "    python discover_petri_nets.py path/to/log_dataset.xes")
            return

        print(f"Found {len(found)} case study log(s) under "
              f"'{base_dir / 'case_studies'}':")
        for dataset_name, xes_path in found:
            print(f"  - {dataset_name}: {xes_path}")
        print()

        jobs = found

    for dataset_name, xes_path in jobs:
        if not xes_path.exists():
            print(f"File not found, skipping: {xes_path}")
            continue
        process_log_file(str(xes_path), args.n_trials, dataset_name=dataset_name)

    print("Done.")


if __name__ == "__main__":
    main()