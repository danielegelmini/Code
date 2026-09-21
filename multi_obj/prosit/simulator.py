import random
import pandas as pd
import math
import heapq
import bisect
import time
from collections import deque
from datetime import datetime
from tqdm import tqdm

from typing import Union, List, Dict, Optional

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

from prosit.discovery.cf_discovery import discover_weight_transitions
from prosit.discovery.time_discovery import discover_execution_time_distributions, discover_arrival_time, discover_waiting_time
from prosit.discovery.calendar_discovery import discover_res_calendars, discover_arrival_calendar
from prosit.discovery.resource_discovery import discover_resources_list, return_multitasking_resources, discover_resource_acts_prob, discover_resources_per_act, discover_weight_resources
from prosit.discovery.data_discovery import discover_attributes_distribution, return_label_data_attributes
from prosit.discovery.online_discovery.cf_discovery import incremental_transition_weights_learning
from prosit.discovery.online_discovery.time_discovery import incremental_execution_time_learning, incremental_model_arrival_learning, incremental_waiting_time_learning
from prosit.utils.common_utils import (
    return_enabled_transitions,
    update_current_marking,
    return_fired_transition,
    count_concurrent_events,
    count_concurrent_events_fast,
    compute_transition_weights_from_model,
    add_minutes_with_calendar,
    build_df_features,
    return_resource,
    )
from prosit.utils.distribution_utils import sampling_from_dist
from prosit.utils.save_and_load_utils import decision_rules_to_dict, transition_to_name, convert_calendar_names, dict_to_decrules, name_to_transition, fromstr_to_scipy

import json
import os


class SimulatorParameters:
    """

    Simulation Parameters
    
    """

    def __init__(
            self, 
            net: PetriNet, 
            initial_marking: Marking,
            final_marking: Marking
        ):
        """
        Creates a SimulatorParameters holder wrapping a given Petri net, and initializes
        every simulation parameter (transition weights, resources, calendars, execution/
        waiting/arrival time distributions) to a trivial, history-independent default (a
        single "auto" resource always available, uniform transition weights, fixed
        distributions all equal to 1). These defaults are placeholders meant to be
        overwritten either by fitting them from a real event log (discover_from_eventlog)
        or by loading a previously-fitted set of parameters (from_json / from_dict).

        Args:
            net (PetriNet): the discovered/loaded Petri net the simulation will run on.
            initial_marking (Marking): the net's initial marking (where every case starts).
            final_marking (Marking): the net's final marking (a case is complete once its
                marking reaches this).

        Returns:
            None (constructor).
        """

        self.net: PetriNet = net
        self.initial_marking: Marking = initial_marking
        self.final_marking: Marking = final_marking
        self.net_transition_labels: list = list(set([t.label for t in net.transitions if t.label]))

        self.label_data_attributes: list = []
        self.label_data_attributes_categorical: list = []
        self.attribute_values_label_categorical: dict = dict()

        self.transition_weights: dict = {t: 1 for t in list(self.net.transitions)}
        self.resources: list = ['auto']
        self.act_resource_prob: dict = {act: {"auto": 1} for act in self.net_transition_labels}
        self.multitasking_resources: list = []
        self.calendars: dict = {'auto': {wd: {h: True for h in range(24)} for wd in range(7)}}
        self.arrival_calendar: dict = {wd: {h: True for h in range(24)} for wd in range(7)}

        self.execution_time_distributions: dict = {a: ('fixed', 1, 1, 1, 1) for a in self.net_transition_labels}
        self.arrival_time_distribution: tuple = ('fixed', 1, 1, 1, 1)
        self.waiting_time_distributions: dict = {'auto': ('fixed', 1, 1, 1, 1)}

        self.rules_mode: bool = False

    @staticmethod
    def _sanitize_for_json(value):
        """
        Recursively converts an arbitrary (possibly nested) Python value into something
        json.dump can serialize: dict keys are passed through _serialize_key (to handle
        non-string keys such as tuples or PetriNet.Transition objects), numpy-like objects
        exposing .tolist() are converted to plain lists, and dicts/lists/tuples are walked
        recursively. Used as the final pass in to_dict() before writing to JSON.

        Args:
            value: any Python value (dict, list, tuple, numpy array/scalar, or a plain
                JSON-native primitive).

        Returns:
            The same structure with every non-JSON-safe piece converted to a JSON-safe one.
        """
        if isinstance(value, dict):
            return {
                SimulatorParameters._serialize_key(k): SimulatorParameters._sanitize_for_json(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SimulatorParameters._sanitize_for_json(v) for v in value]
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                return value
        return value

    @staticmethod
    def _serialize_key(key):
        """
        Converts a single dict key into a JSON-safe key. JSON object keys must be plain
        strings, so a tuple key (e.g. a (case_id, prefix) style key) is encoded into a
        special "__tuple__:<json-encoded tuple>" string that _restore_from_json later
        recognizes and decodes back into a real tuple. Ordinary JSON-native primitives
        (str/int/float/bool/None) pass through unchanged; anything else (e.g. a
        PetriNet.Transition object used as a dict key) falls back to str(key).

        Args:
            key: any hashable value used as a dict key.

        Returns:
            A JSON-safe key: the original value if already JSON-native, otherwise a string.
        """
        if isinstance(key, tuple):
            return f"__tuple__:{json.dumps(key)}"
        if isinstance(key, (str, int, float, bool)) or key is None:
            return key
        return str(key)

    @staticmethod
    def _restore_from_json(value):
        """
        Inverse of _sanitize_for_json / _serialize_key: walks a dict/list structure just
        loaded from JSON and turns any dict key that starts with the "__tuple__:" marker
        back into a real Python tuple (by parsing the JSON payload embedded after the
        marker), recursively over the whole structure. Called once at the top of
        from_dict() before any parameter is actually read out of dict_params.

        Args:
            value: a dict/list/primitive, as produced by json.load.

        Returns:
            The same structure with every "__tuple__:..." key restored to a tuple key.
        """
        if isinstance(value, dict):
            restored = {}
            for k, v in value.items():
                if isinstance(k, str) and k.startswith("__tuple__:"):
                    restored[tuple(json.loads(k[len("__tuple__:"):]))] = SimulatorParameters._restore_from_json(v)
                else:
                    restored[k] = SimulatorParameters._restore_from_json(v)
            return restored
        if isinstance(value, list):
            return [SimulatorParameters._restore_from_json(v) for v in value]
        return value

    @staticmethod
    def _normalize_distribution_params(params):
        """
        Normalizes a distribution's "params" field (as read back from a loaded JSON file)
        into a tuple, regardless of whether it was stored as None, a list, or a single
        scalar value. Needed because downstream code passes these params positionally into
        a scipy.stats distribution constructor (dist(*params)), which requires a tuple/list,
        not an arbitrary type.

        Args:
            params: None, a list/tuple of distribution parameters, or a single scalar.

        Returns:
            tuple: () if params was None, tuple(params) if it was already list-like,
            otherwise a 1-element tuple wrapping the scalar.
        """
        if params is None:
            return ()
        if isinstance(params, (list, tuple)):
            return tuple(params)
        return (params,)

    def discover_from_eventlog(
            self, 
            log: EventLog, 
            max_depth_tree: int = 3,
            incremental_discovery: bool = False,
            grace_period: int = 1000,
            verbose: bool = True
        ):
        """
        Fits every simulation parameter of this SimulatorParameters instance from a real
        event log, by delegating to the discovery functions in prosit/discovery/ and
        prosit/discovery/online_discovery/: control-flow / transition weights
        (discover_weight_transitions, one binary "did this transition fire when enabled"
        model per transition -- see cf_discovery.py), execution time, waiting time and
        arrival time distributions, plus resources, calendars and (if present) case/event
        data-attribute distributions. Mutates self in place; does not return anything.

        Two independent axes control HOW each parameter is fit:
        - max_depth_tree selects history-conditioned "rules mode" (a decision tree per
          transition/activity/resource, cross-validated over depths 1..max_depth_tree,
          self.rules_mode=True) when >= 1, or flat, history-independent probabilities/
          distributions (self.rules_mode=False, no trees at all) when < 1.
        - incremental_discovery selects streaming/online fitting (river's
          HoeffdingAdaptiveTreeClassifier, one pass over the log row by row) instead of the
          default batch fitting (scikit-learn, sees the whole dataset at once, with
          cross-validation over max_depth_tree candidate depths).

        Any transition never found enabled anywhere in the (aligned) log is defaulted to a
        constant weight of 0 after discovery, since it has no training data to fit a
        probability from.

        Args:
            log (EventLog): the (training) event log to discover parameters from.
            max_depth_tree (int, optional): maximum decision-tree depth for control-flow/
                execution/waiting-time trees; < 1 disables trees entirely (flat mode).
                Defaults to 3.
            incremental_discovery (bool, optional): if True, fit every model incrementally/
                online (river) instead of in one batch (scikit-learn). Defaults to False.
            grace_period (int, optional): river's minimum number of observations an
                incremental tree waits before considering a split (only used when
                incremental_discovery=True). Defaults to 1000.
            verbose (bool, optional): if True, print a progress message before each
                discovery stage. Defaults to True.

        Returns:
            None (mutates self in place: transition_weights, execution_time_distributions,
            waiting_time_distributions, arrival_time_distribution, resources,
            act_resource_prob, multitasking_resources, calendars, arrival_calendar,
            label_data_attributes(_categorical), attribute_values_label_categorical,
            distribution_data_attributes, rules_mode).
        """

        if max_depth_tree < 1:
            self.rules_mode = False
            max_depth_cv = []
        else:
            self.rules_mode = True
            max_depth_cv = range(1, max_depth_tree + 1)
        
        self.label_data_attributes, self.label_data_attributes_categorical = return_label_data_attributes(log)
        
        for a in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        if verbose:
            print("Resources discovery...")
        self.resources = discover_resources_list(log, thr=1.0)
        self.act_resource_prob = discover_resource_acts_prob(log, self.resources)

        if self.label_data_attributes:
            if verbose:
                print("Data attributes discovery...")
            self.distribution_data_attributes = discover_attributes_distribution(log, self.label_data_attributes)
        else:
            self.distribution_data_attributes = None

        if verbose:
            print("Feature discovery...")
        df_features = build_df_features(log, self.net, self.initial_marking, self.final_marking, self.act_resource_prob, self.net_transition_labels, self.resources, self.label_data_attributes)
        df_features = df_features[(df_features['resource'].isin(self.resources)) | (df_features["resource"].isna())]
        df_features.reset_index(drop=True, inplace=True)
        self.multitasking_resources = return_multitasking_resources(df_features)

        if verbose:
            if incremental_discovery:
                print("Incremental Transition Probabilities discovery...")
            else:
                print("Transition Probabilities discovery...")
        
        if incremental_discovery:
            self.transition_weights = incremental_transition_weights_learning(
                                                                    df_features, 
                                                                    self.net_transition_labels, 
                                                                    max_depth=max_depth_tree,
                                                                    grace_period=grace_period,                  
                                                                    label_data_attributes=self.label_data_attributes, 
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                    values_categorical=self.attribute_values_label_categorical
                                                                )
        else:
            self.transition_weights = discover_weight_transitions(
                                                                    df_features, 
                                                                    self.net_transition_labels, 
                                                                    max_depths_cv=max_depth_cv,                  
                                                                    label_data_attributes=self.label_data_attributes, 
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                    values_categorical=self.attribute_values_label_categorical
                                                                )
        for t in self.net.transitions:
            if t not in self.transition_weights.keys():
                self.transition_weights[t] = 0

        if verbose:
            print("Calendars discovery...")
        self.calendars = discover_res_calendars(log, self.resources)
        self.arrival_calendar = discover_arrival_calendar(log)

        if verbose:
            if incremental_discovery:
                print("Incremental Execution Time discovery...")
            else:
                print("Execution Time discovery...")

        if incremental_discovery:
            self.execution_time_distributions = incremental_execution_time_learning(    
                                                                                        df_features,
                                                                                        self.net_transition_labels,
                                                                                        self.resources,
                                                                                        self.calendars, 
                                                                                        max_depth=max_depth_tree,
                                                                                        grace_period=grace_period,
                                                                                        label_data_attributes=self.label_data_attributes, 
                                                                                        label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                                        values_categorical=self.attribute_values_label_categorical
                                                                                    )
        else:
            self.execution_time_distributions = discover_execution_time_distributions(
                                                                                        df_features,
                                                                                        self.net_transition_labels,
                                                                                        self.resources,
                                                                                        self.calendars, 
                                                                                        max_depths=max_depth_cv,
                                                                                        label_data_attributes=self.label_data_attributes, 
                                                                                        label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                                        values_categorical=self.attribute_values_label_categorical
                                                                                    )
        if verbose:
            if incremental_discovery:
                print("Incremental Waiting Time discovery...")
            else:
                print("Waiting Time discovery...")

        if incremental_discovery:
            self.waiting_time_distributions = incremental_waiting_time_learning(
                                                                                    df_features,
                                                                                    self.net_transition_labels,
                                                                                    self.resources, 
                                                                                    self.calendars, 
                                                                                    self.label_data_attributes, 
                                                                                    self.label_data_attributes_categorical, 
                                                                                    self.attribute_values_label_categorical, 
                                                                                    max_depth=max_depth_tree,
                                                                                    grace_period=grace_period
                                                                                )
        else:
            self.waiting_time_distributions = discover_waiting_time(
                                                                        df_features,
                                                                        self.net_transition_labels,
                                                                        self.resources, 
                                                                        self.calendars, 
                                                                        self.label_data_attributes, 
                                                                        self.label_data_attributes_categorical, 
                                                                        self.attribute_values_label_categorical, 
                                                                        max_depths=max_depth_cv
                                                                    )
        
        if verbose:
            if incremental_discovery:
                print("Incremental Arrival Time discovery...")
            else:
                print("Arrival Time discovery...")
        
        if incremental_discovery:
            self.arrival_time_distribution = incremental_model_arrival_learning(log, self.arrival_calendar, max_depth=max_depth_tree, grace_period=grace_period)
        else:
            self.arrival_time_distribution = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depth_cv)


    def to_dict(self) ->  dict:
        """
        Serializes every fitted parameter of this instance into a plain, nested dict of
        JSON-safe primitives -- the exact structure from_dict() later reads back. Converts
        PetriNet.Transition objects to plain names (transition_to_name), DecisionRules trees
        to nested dicts (decision_rules_to_dict), and resource calendars to a JSON-friendly
        weekday/hour encoding (convert_calendar_names), then runs the whole result through
        _sanitize_for_json as a final safety pass (handles remaining non-string dict keys,
        numpy types, etc.).

        Args:
            None (self).

        Returns:
            dict: nested parameter dict, ready to be passed to json.dump (see to_json).
        """

        dict_params = {

            "transition_params": {
                "transition_weights": {transition_to_name(t): decision_rules_to_dict(dr) for t, dr in self.transition_weights.items()} # ok
                },

            "resource_params": {
                "resources" : self.resources,
                "resource_probabilities": self.act_resource_prob,
                "multitasking_resource": self.multitasking_resources,
                "calendars": {r: convert_calendar_names(cal) for r, cal in self.calendars.items()}
                },

            "arrival_params": {
                "arrival_calendar": convert_calendar_names(self.arrival_calendar),
                "arrival_time_distributions": decision_rules_to_dict(self.arrival_time_distribution)
                },

            "execution_time_params": {
                "execution_time_distributions": {a: decision_rules_to_dict(dr) for a, dr in self.execution_time_distributions.items()} 
                },

            "waiting_time_params": {
                "waiting_time_distributions": {r: decision_rules_to_dict(dr) for r, dr in self.waiting_time_distributions.items()} 
                },

            "data_attribute_params": {
                "label_data_attributes": self.label_data_attributes,
                "label_data_attributes_categorical": self.label_data_attributes_categorical,
                "attribute_values_label_categorical": self.attribute_values_label_categorical,
                "distribution_data_attributes": self.distribution_data_attributes
                }

        }

        return self._sanitize_for_json(dict_params)

    def to_json(self, path: str = "simulator_params.json"):
        """
        Writes self.to_dict() to disk as JSON at `path`. Writes to a temporary "<path>.tmp"
        file first and only then atomically replaces `path` with it (os.replace), so a
        crash or interruption mid-write can never leave a corrupted/partially-written
        parameters file at the real destination path.

        Args:
            path (str, optional): destination file path. Defaults to "simulator_params.json".

        Returns:
            None (side effect: writes/overwrites the file at `path`).
        """

        dict_params = self.to_dict()
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump(dict_params, json_file, indent=4)
        os.replace(temp_path, path)


    def from_dict(self, dict_params):
        """
        Inverse of to_dict(): restores every simulation parameter attribute of this instance
        from a plain nested dict (as produced by to_dict(), typically after a json.load).
        Requires self.net to already be a valid, matching PetriNet (transition names are
        mapped back to real PetriNet.Transition objects via name_to_transition(t_name,
        self.net)).

        self.rules_mode is inferred from the shape of the stored arrival-time distribution
        (whether it has a "mean_value" key, which only flat/non-tree distributions carry),
        and every other attribute is then restored either as DecisionRules trees
        (rules_mode=True, via dict_to_decrules) or as flat (scipy_dist, params, min, max,
        mean) tuples (rules_mode=False).

        Args:
            dict_params (dict): nested parameter dict, as produced by to_dict() (or loaded
                from JSON via from_json).

        Returns:
            None (mutates self in place: rules_mode, label_data_attributes(_categorical),
            attribute_values_label_categorical, distribution_data_attributes, resources,
            act_resource_prob, multitasking_resources, calendars, arrival_calendar,
            transition_weights, execution_time_distributions, waiting_time_distributions,
            arrival_time_distribution).
        """

        dict_params = self._restore_from_json(dict_params)

        self.rules_mode = "mean_value" not in dict_params["arrival_params"]["arrival_time_distributions"].keys()
        self.label_data_attributes, self.label_data_attributes_categorical = dict_params["data_attribute_params"]["label_data_attributes"], dict_params["data_attribute_params"]["label_data_attributes_categorical"]
        self.attribute_values_label_categorical = dict_params["data_attribute_params"]["attribute_values_label_categorical"]
        self.distribution_data_attributes = dict_params["data_attribute_params"]["distribution_data_attributes"]

        self.resources = dict_params["resource_params"]["resources"]
        self.act_resource_prob = dict_params["resource_params"]["resource_probabilities"]
        self.multitasking_resources = dict_params["resource_params"]["multitasking_resource"]

        self.calendars = {r: convert_calendar_names(cal, to_number=True) for r, cal in dict_params["resource_params"]["calendars"].items()}
        self.arrival_calendar = convert_calendar_names(dict_params["arrival_params"]["arrival_calendar"], to_number=True)

        if self.rules_mode:
            self.transition_weights = {name_to_transition(t_name, self.net): dict_to_decrules(value) for t_name, value in dict_params["transition_params"]["transition_weights"].items()}
            self.execution_time_distributions = {act: dict_to_decrules(value) for act, value in dict_params["execution_time_params"]["execution_time_distributions"].items()}
            self.waiting_time_distributions = {res: dict_to_decrules(value) for res, value in dict_params["waiting_time_params"]["waiting_time_distributions"].items()}
            self.arrival_time_distribution = dict_to_decrules(dict_params["arrival_params"]["arrival_time_distributions"])
        else:
            self.transition_weights = {name_to_transition(t_name, self.net): value for t_name, value in dict_params["transition_params"]["transition_weights"].items()}  
            self.execution_time_distributions = {
                act: (
                    fromstr_to_scipy(value["dist_name"]),
                    self._normalize_distribution_params(value.get("params")),
                    value["min_value"],
                    value["max_value"],
                    value["mean_value"],
                )
                for act, value in dict_params["execution_time_params"]["execution_time_distributions"].items()
            }
            self.waiting_time_distributions = {
                res: (
                    fromstr_to_scipy(value["dist_name"]),
                    self._normalize_distribution_params(value.get("params")),
                    value["min_value"],
                    value["max_value"],
                    value["mean_value"],
                )
                for res, value in dict_params["waiting_time_params"]["waiting_time_distributions"].items()
            }
            arrival_dist = dict_params["arrival_params"]["arrival_time_distributions"]
            self.arrival_time_distribution = (
                fromstr_to_scipy(arrival_dist["dist_name"]),
                self._normalize_distribution_params(arrival_dist.get("params")),
                arrival_dist["min_value"],
                arrival_dist["max_value"],
                arrival_dist["mean_value"],
            )

    def from_json(self, path: str = "simulator_params.json"):
        """
        Loads simulation parameters from a JSON file on disk (as written by to_json()) and
        populates this instance via from_dict(). Raises a clear, actionable ValueError
        instead of letting a raw json.JSONDecodeError propagate if the file is corrupted or
        not valid JSON -- typically the sign of a discovery run that was interrupted before
        finishing to_json()'s write (though to_json()'s own atomic write greatly reduces how
        often that can actually happen).

        Args:
            path (str, optional): source file path. Defaults to "simulator_params.json".

        Returns:
            None (mutates self in place via from_dict()).

        Raises:
            ValueError: if the file at `path` is not valid JSON.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                dict_params = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Simulator parameters cache is corrupted or invalid JSON: {path}. "
                "Remove the file and rerun discovery."
            ) from exc
        self.from_dict(dict_params)



class SimulatorEngine:

    def __init__(
            self, 
            simulation_parameters: SimulatorParameters
        ):
        """
        Wraps an already-fitted SimulatorParameters into a runnable simulation engine.
        Precomputes a fixed, deterministic ordering over every transition in the net
        (_build_transition_rank -- used by _sort_transitions/_get_enabled_transitions_sorted
        so that, whenever multiple transitions are enabled together, they are always
        presented to the rest of the code in the same order across runs, keeping simulation
        output reproducible instead of depending on incidental Python set/dict iteration
        order) and initializes the diagnostic lists and caches populated during apply() runs
        (last_unreachable_recommendations, last_runaway_cases, last_non_fitting_prefixes,
        last_model_inserted_activities, _prefix_state_cache -- see their own inline comments
        below and apply()'s docstring for what each one records).

        Args:
            simulation_parameters (SimulatorParameters): a parameters object already fitted
                via discover_from_eventlog(), or restored via from_json()/from_dict().

        Returns:
            None (constructor).
        """

        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters
        self._transition_rank = self._build_transition_rank()
        self.raise_on_unreachable_recommendation = False
        self.last_unreachable_recommendations = []
        # Cases that were force-truncated because they exceeded max_events_per_case
        # (safety valve against runaway/near-infinite loops -- see apply()).
        self.last_runaway_cases = []
        # Historical prefixes whose alignment-based replay required at least one "log move"
        # (a logged activity the model could not explain at all -- a genuine deviation).
        # Recorded here for visibility only; it does not affect the reconstructed marking's
        # legality -- see _reconstruct_prefix_state.
        self.last_non_fitting_prefixes = []
        # Historical prefixes whose alignment-based replay needed to insert at least one
        # visible activity that is NOT present in the historical log at all -- the model
        # considers it a necessary step to get from one logged activity to the next, but it
        # was never actually recorded. Never written to the output log (the saved prefix is
        # always exactly the input log, unchanged); it does affect the case's activity-history
        # counts used for probabilistic weighting of subsequent simulated events -- see
        # _reconstruct_prefix_state.
        self.last_model_inserted_activities = []
        # Cache of _reconstruct_prefix_state results, keyed by (case id, prefix activity
        # sequence). Alignment computation is not free, and the same historical prefix is
        # replayed identically every time apply() is called with it -- e.g. across the N
        # runs of a batch, or across baseline/exhaustive/nsga2 for the same underlying case
        # (recommendation:act/res differ, but the prefix's own activity sequence does not).
        # Keyed by content, not by object identity, so it stays correct even if the caller
        # passes a freshly-rebuilt DataFrame each time.
        self._prefix_state_cache = {}
        # Node/time cap for _bfs_path_to_activity -- see apply()'s
        # max_reachability_search_nodes / max_reachability_search_seconds.
        self._reachability_search_max_nodes = 30000
        self._reachability_search_max_seconds = 5.0

    def _build_transition_rank(self) -> dict:
        """
        Builds a fixed, deterministic total ordering over every transition in the net,
        sorted by label, then internal transition name, then input/output place names (as
        a tie-breaker for transitions sharing the same label, e.g. duplicate/silent-routing
        copies of the same activity). Called once in __init__(); the resulting lookup is
        what _sort_transitions uses to always present a given set of transitions in the
        same order, regardless of incidental Python set/dict iteration order.

        Args:
            None (self, reads self.net.transitions).

        Returns:
            dict: {PetriNet.Transition: int rank}, rank 0 being first in the fixed order.
        """
        ordered = sorted(
            list(self.net.transitions),
            key=lambda t: (
                str(t.label) if t.label is not None else "",
                str(getattr(t, "name", "")),
                tuple(sorted(str(a.source) for a in t.in_arcs)),
                tuple(sorted(str(a.target) for a in t.out_arcs)),
            ),
        )
        return {t: i for i, t in enumerate(ordered)}

    def _sort_transitions(self, transitions) -> list:
        """
        Sorts an arbitrary iterable of transitions into the engine's fixed deterministic
        order (via the ranks precomputed by _build_transition_rank; anything not found in
        the rank table -- should not normally happen -- sorts last, by label/name as a
        fallback key).

        Args:
            transitions (iterable of PetriNet.Transition): the transitions to sort.

        Returns:
            list of PetriNet.Transition: the same transitions, deterministically ordered.
        """
        return sorted(
            list(transitions),
            key=lambda t: (
                self._transition_rank.get(t, math.inf),
                str(t.label) if t.label is not None else "",
                str(getattr(t, "name", "")),
            ),
        )

    def _get_enabled_transitions_sorted(self, marking) -> list:
        """
        Convenience wrapper: computes which transitions are enabled at a given Petri net
        marking (return_enabled_transitions, the token-game check) and returns them in the
        engine's fixed deterministic order (_sort_transitions) instead of raw set order.
        This is the function called everywhere in the simulation loop and in the discovery/
        reachability helpers whenever "what can fire from here" is needed.

        Args:
            marking (Marking): the Petri net marking to check.

        Returns:
            list of PetriNet.Transition: every transition enabled at `marking`, in the
            engine's fixed deterministic order.
        """
        return self._sort_transitions(return_enabled_transitions(self.net, marking))

    def _get_case_enabled_time(self, case) -> datetime:
        """
        Returns the earliest timestamp among a case's currently-enabled transitions (i.e.
        the next moment this case is ready to fire something), or -- if nothing is enabled
        yet (e.g. right at the very start, before "enabled" has been populated) -- falls
        back to the case's arrival time. Used as a default "enabled at" timestamp in a couple
        of places where no more specific one is available.

        Args:
            case (dict): one entry of apply()'s internal `cases` list (must have "enabled"
                and "arrival_time" keys).

        Returns:
            datetime: the case's next enabled time, or its arrival time as a fallback.
        """
        if case["enabled"]:
            return min(case["enabled"].values())
        return case["arrival_time"]

    def _bfs_path_to_activity(self, start_marking, target_activity, only_invisible, max_depth=200):
        """
        BFS for a firing sequence from start_marking after which target_activity becomes
        enabled (the returned path does not include firing target_activity itself).

        Bounded by max_depth (path length) AND by self._reachability_search_max_nodes /
        self._reachability_search_max_seconds (distinct markings visited / wall clock).
        The latter two exist because a case seeded from a non-fitting historical prefix
        (see last_non_fitting_prefixes) can carry a marking whose reachable state space
        explodes combinatorially when the net has several concurrent branches -- without
        a bound this search can run for a very long time (observed: 70k+ markings visited,
        10+ seconds, on BPI12) even though it is technically finite. When the cap is hit
        the outcome is genuinely undetermined, which is reported distinctly from a search
        that fully exhausted the reachable state space without finding the target (that
        one really is unreachable).

        Args:
            start_marking (Marking): the marking to search from.
            target_activity (str or None): the activity label to look for (a transition
                becomes the search's target as soon as it is enabled with this label).
            only_invisible (bool): if True, only invisible/silent transitions (label is
                None) may be fired while searching -- used to find a "free" routing path
                that doesn't consume/emit any visible activity of its own. If False, any
                enabled transition (visible or not) may be used to reach the target.
            max_depth (int, optional): maximum path length (number of transitions fired)
                to explore. Defaults to 200.

        Returns:
            tuple(path_or_None, status): `path` is a list of PetriNet.Transition to fire
            (in order) to enable target_activity, not including target_activity itself, or
            None if no path was found within the given bounds. `status` is one of "found",
            "exhausted" (the full reachable state space was searched, target truly
            unreachable), or "capped" (the max_nodes/max_seconds safety limit was hit before
            either finding the target or exhausting the state space -- reachability is
            genuinely undetermined in this case, not proven impossible).
        """
        queue = deque([(start_marking, [])])
        visited = {frozenset(start_marking.items())}
        t0 = time.monotonic()
        max_nodes = getattr(self, "_reachability_search_max_nodes", 30000)
        max_seconds = getattr(self, "_reachability_search_max_seconds", 5.0)

        while queue:
            if len(visited) > max_nodes or (time.monotonic() - t0) > max_seconds:
                return None, "capped"

            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            enabled = self._get_enabled_transitions_sorted(marking)
            if any(t.label == target_activity for t in enabled):
                return path, "found"

            for t in enabled:
                if only_invisible and t.label is not None:
                    continue
                next_marking = update_current_marking(marking, t)
                key = frozenset(next_marking.items())
                if key not in visited:
                    visited.add(key)
                    queue.append((next_marking, path + [t]))

        return None, "exhausted"

    def _reconstruct_prefix_state(self, case_id_c, prefix_log_c_sorted):
        """
        Reconstructs the Petri net marking and per-activity firing counts after replaying a
        case's historical prefix, using cost-based alignment instead of naive token-based
        replay.

        Why: token-based replay, when a trace does not perfectly fit the model, patches over
        the mismatch by inserting "missing tokens" wherever a logged activity is not
        structurally enabled. The resulting marking can then contain tokens that are not
        actually reachable from the initial marking via any legal firing sequence -- observed
        on BPI12: extra tokens stranded in a loop-join place, permanently disabling that
        loop's only exit and turning what should be a normal rework loop into a genuine,
        unrecoverable deadlock for that case (no amount of extra simulated events can ever
        complete it). Since the net is a sound workflow net (the "option to complete"
        property holds for every reachable marking), a marking reconstructed via alignment
        -- which only ever takes legal model moves, never invents tokens -- is GUARANTEED to
        be able to reach the final marking. Verified empirically on all 633 BPI12 prefixes:
        0 cases where the final marking became unreachable after this change (was 1 case
        hitting the max_events_per_case safety valve under token-based replay).

        How: pm4py's alignment always targets the net's actual final marking, which for a
        partial prefix means it also inserts "model moves" to force completion of the whole
        remaining process -- not what we want here, we only want the state after the prefix.
        So the alignment is truncated right after its last move that corresponds to a real
        prefix event; everything after that (the forced-completion tail) is discarded. The
        (truncated) sequence of required visible activities is then replayed one at a time:
        if not already enabled, _bfs_path_to_activity (invisible-only) finds the legal path
        to enable it -- the same mechanism already used for recommendation reachability.

        Cached per (case id, exact prefix activity sequence) since alignment is not free and
        the same historical prefix is replayed identically every time apply() is called with
        it (e.g. across the N runs of a batch, or across baseline/exhaustive/nsga2 for the
        same underlying case).

        Args:
            case_id_c: the case's identifier (used only as part of the cache key).
            prefix_log_c_sorted (pd.DataFrame): this case's historical prefix event rows,
                sorted chronologically, with a "concept:name" column giving each event's
                activity label.

        Returns:
            dict with keys:
                "marking" (Marking): the reconstructed Petri net marking right after the
                    prefix (guaranteed legally reachable from the initial marking).
                "history" (dict): {activity_label: count} of how many times each activity
                    fired to reach this marking (including any model-inserted activities --
                    see below), used as a feature for history-conditioned probabilistic
                    weighting of subsequent simulated events.
                "is_fit" (bool): False if the alignment needed at least one "log move" (a
                    logged activity the model could not explain at all) to explain the
                    prefix; True if every logged event was matched by a real model move.
                "inserted_activities" (list of str): visible activity labels the alignment
                    fired against the net to explain the prefix but which are NOT present in
                    the historical log itself (see the inline comment above where this list
                    is built) -- affects "history" but is never written to the output log.
        """
        cache_key = (case_id_c, tuple(prefix_log_c_sorted["concept:name"]))
        cached = self._prefix_state_cache.get(cache_key)
        if cached is not None:
            return cached

        trace = Trace()
        for act in prefix_log_c_sorted["concept:name"]:
            trace.append(Event({"concept:name": act}))

        align_result = alignments.apply_trace(
            trace, self.net, self.initial_marking, self.final_marking,
            parameters={alignments.Parameters.ACTIVITY_KEY: "concept:name"},
        )

        moves = align_result["alignment"]
        log_move_indices = [i for i, (log_move, _) in enumerate(moves) if log_move != ">>"]
        last_log_idx = max(log_move_indices) if log_move_indices else -1
        relevant_moves = moves[:last_log_idx + 1]

        is_fit = all(model_move != ">>" for log_move, model_move in relevant_moves if log_move != ">>")
        required_visible_labels = [
            model_move for _, model_move in relevant_moves if model_move not in (">>", None)
        ]
        # Visible activities the model needed to explain the prefix that are NOT present in
        # the historical log at all (log_move == ">>"): the model considers them necessary to
        # get from one logged activity to the next, but they were never actually recorded.
        # These are fired against the net (they affect the marking and the history counts
        # used for probabilistic weighting) but are NEVER written to the output log -- the
        # saved prefix is always exactly what was in the input log, unchanged.
        inserted_activities = [
            model_move for log_move, model_move in relevant_moves
            if log_move == ">>" and model_move not in (">>", None)
        ]

        marking = self.initial_marking
        history = {t: 0 for t in self.simulation_parameters.net_transition_labels}
        for label in required_visible_labels:
            enabled = self._get_enabled_transitions_sorted(marking)
            labels = [t.label for t in enabled]
            if label not in labels:
                invisible_path, _ = self._bfs_path_to_activity(marking, label, only_invisible=True)
                if invisible_path is None:
                    # Should not happen if the alignment is valid and the net matches --
                    # defensively stop here rather than raise, leaving a still-legal (if
                    # earlier-than-expected) marking.
                    break
                for invisible_t in invisible_path:
                    marking = update_current_marking(marking, invisible_t)
                enabled = self._get_enabled_transitions_sorted(marking)
                labels = [t.label for t in enabled]
                if label not in labels:
                    break
            chosen = enabled[labels.index(label)]
            marking = update_current_marking(marking, chosen)
            history[label] += 1

        result = {
            "marking": marking,
            "history": history,
            "is_fit": is_fit,
            "inserted_activities": inserted_activities,
        }
        self._prefix_state_cache[cache_key] = result
        return result

    def _resolve_recommended_transition(self, case, enabled_transitions):
        """
        Decides how to advance a case that has a pending recommendation (case["rec_act"]/
        case["rec_res"]) for this simulation step, trying to realize that recommendation as
        the very next (or next reachable) step, best-effort, in this priority order:
        1. If a multi-step path to the recommendation was already computed on a previous
           call (case["pending_invisible_path"]), consume its next transition.
        2. If the recommended activity is already directly enabled, fire it immediately.
        3. Otherwise, search for a legal path using ONLY invisible/silent transitions
           (_bfs_path_to_activity, only_invisible=True) to reach it "for free"; if found,
           queue the rest of the path and start consuming it.
        4. Otherwise (and only if step 3's search was not itself capped, since a wider
           search would almost certainly hit the same cap too), search a broader path that
           may also fire visible transitions along the way.
        5. If no path is found at all: in strict mode (case["strict_recommendation"]),
           record the failure in self.last_unreachable_recommendations (and raise if
           self.raise_on_unreachable_recommendation is set); either way, give up on the
           recommendation (clear rec_act/rec_res) so the caller falls back to normal
           weighted simulation for this step.

        A synthetic sentinel recommendation (case["rec_act"] starting with
        "__NO_RECOMMENDATION__", used by baseline simulation runs) is treated as "no
        recommendation" immediately, without triggering any search.

        Args:
            case (dict): the case being advanced (mutated in place: may update
                case["pending_invisible_path"], case["rec_act"]/case["rec_res"]).
            enabled_transitions (list of PetriNet.Transition): transitions currently enabled
                at case["marking"] (used for the direct-match check in step 2).

        Returns:
            tuple(chosen_transition, activity, t_enabled, flag_rec):
                chosen_transition (PetriNet.Transition or None): the transition to fire this
                    step, or None if no recommendation-driven move was made.
                activity (str or None): its label, if directly matching the recommendation
                    (None for an intermediate invisible/path step, even if chosen_transition
                    is not None).
                t_enabled (datetime or None): when it became enabled.
                flag_rec (bool): True if this step was recommendation-driven (the caller
                    should skip the normal weighted-roulette choice), False otherwise.
        """
        if case["rec_act"] is None:
            return None, None, None, False

        # Baseline runs use a synthetic sentinel recommendation that should not
        # trigger any reachability search.
        if isinstance(case["rec_act"], str) and case["rec_act"].startswith("__NO_RECOMMENDATION__"):
            case["rec_act"] = None
            case["rec_res"] = None
            return None, None, None, False

        pending_path = case.get("pending_invisible_path", [])
        if pending_path:
            chosen_transition = pending_path[0]
            case["pending_invisible_path"] = pending_path[1:]
            return chosen_transition, None, case["enabled"].get(chosen_transition, self._get_case_enabled_time(case)), True

        enabled_transitions = self._sort_transitions(enabled_transitions)
        enabled_transitions_labels = [t.label for t in enabled_transitions]
        if case["rec_act"] in enabled_transitions_labels:
            chosen_transition = enabled_transitions[enabled_transitions_labels.index(case["rec_act"])]
            return chosen_transition, case["rec_act"], case["enabled"][chosen_transition], True

        invisible_path, invisible_status = self._bfs_path_to_activity(case["marking"], case["rec_act"], only_invisible=True)
        if invisible_path is not None:
            chosen_transition = invisible_path[0]
            case["pending_invisible_path"] = invisible_path[1:]
            return chosen_transition, None, case["enabled"].get(chosen_transition, self._get_case_enabled_time(case)), True

        # If invisible-only path does not exist, walk the deterministic shortest
        # path until recommendation becomes enabled while still emitting the
        # recommendation as the first post-prefix visible activity. Skipped when
        # the invisible-only search already hit the node/time cap: the full search
        # (which allows strictly more transitions per step, so branches even wider)
        # would almost certainly hit the same cap too, for no benefit.
        reach_path, reach_status = (None, invisible_status)
        if invisible_status != "capped":
            reach_path, reach_status = self._bfs_path_to_activity(case["marking"], case["rec_act"], only_invisible=False)
            if reach_path is not None:
                chosen_transition = reach_path[0]
                case["pending_invisible_path"] = reach_path[1:]
                return chosen_transition, None, case["enabled"].get(chosen_transition, self._get_case_enabled_time(case)), True

        # In strict mode, never consume other visible activities before recommendation.
        if case.get("strict_recommendation", False):
            case_external_id = case.get("case_external_id", case.get("case_id"))
            rec_act = case.get("rec_act")
            rec_res = case.get("rec_res")
            if "capped" in (invisible_status, reach_status):
                reason = (
                    "reachability_search_aborted: the search space explored while looking for a "
                    "legal path to the recommended activity exceeded the node/time safety cap "
                    "before finding it or exhausting the state space -- reachability is genuinely "
                    "undetermined (not proven impossible), most likely caused by a non-fitting "
                    "historical prefix (see last_non_fitting_prefixes) seeding a highly concurrent marking."
                )
            else:
                reason = "not_reachable_from_replayed_prefix_marking"
            self.last_unreachable_recommendations.append(
                {
                    "case:concept:name": str(case_external_id),
                    "recommendation:act": rec_act,
                    "recommendation:res": rec_res,
                    "reason": reason,
                }
            )
            if self.raise_on_unreachable_recommendation:
                raise RuntimeError(
                    f"Strict recommendation failed for case {case_external_id}: "
                    f"activity '{rec_act}' is not reachable from the replayed prefix marking."
                )

        case["rec_act"] = None
        case["rec_res"] = None
        return None, None, None, False

    @staticmethod
    def _apply_recommendation_lock(case, enabled_time, transition):
        """
        After a recommended activity has just fired, the case is "locked" (via
        case["recommendation_lock_until"], set in apply() right after firing a
        recommendation-matching transition) so that no OTHER visible activity can start
        before that time -- this stops the normal weighted-random background simulation
        from immediately racing past / overtaking the just-applied recommendation. Only
        visible transitions are ever delayed by the lock; invisible/silent transitions
        (transition.label is None) are structural routing steps and are never held back.

        Args:
            case (dict): the case being evaluated (reads case["recommendation_lock_until"]).
            enabled_time (datetime): the transition's naturally computed enabled time.
            transition (PetriNet.Transition): the transition being considered.

        Returns:
            datetime: `enabled_time` unchanged if no lock applies (no lock set, it has
            already passed, or `transition` is invisible), otherwise
            max(enabled_time, lock_until).
        """
        lock_until = case.get("recommendation_lock_until")
        if lock_until is None:
            return enabled_time
        if transition.label is None:
            return enabled_time
        return max(enabled_time, lock_until)

    def apply(
        self,
        n_traces: int = 1,
        t_start: datetime = datetime.now(),
        deterministic_time: bool = False,
        prev_log: Optional[pd.DataFrame] = None,
        max_events_per_case: Optional[int] = 300,
        max_reachability_search_nodes: int = 30000,
        max_reachability_search_seconds: float = 5.0,
    ) -> pd.DataFrame:
        """
        Runs the discrete-event simulation itself and returns the resulting event log.

        Internally drives a min-heap priority queue of (next_enabled_time, case_id): at each
        iteration it pops whichever case's next event is chronologically earliest across ALL
        cases, decides which transition fires for it (a pending recommendation if one
        applies -- see _resolve_recommended_transition -- otherwise a weighted random draw
        over the case's currently-enabled transitions via return_fired_transition), resolves
        a resource assignment, samples/computes a waiting time and an execution time,
        appends the resulting event, advances that case's Petri net marking, and -- unless
        the case just completed (reached self.final_marking) -- re-enqueues it with its new
        next-enabled time. Repeats until the heap is empty (every case has either completed
        or been force-truncated).

        Two mutually exclusive modes, selected by whether `prev_log` is given:
        - Pure generation (prev_log=None): samples `n_traces` brand-new cases from scratch
          (arrival time, then every event) using the fitted simulation_parameters, with no
          historical data involved.
        - Recommendation / prefix-continuation (prev_log given): `n_traces` is ignored (set
          to 0); instead, every case in `prev_log` carrying a
          recommendation:act/recommendation:res value is treated as a real historical
          prefix. Each such prefix's current Petri net state is reconstructed via alignment
          (_reconstruct_prefix_state), and only the FUTURE of that prefix is simulated,
          best-effort honoring the given recommendation as the next step before falling back
          to normal weighted simulation for everything after it. The returned log then
          contains the original prev_log rows concatenated with the newly simulated
          continuation events.

        Every call resets and repopulates four diagnostic lists on self, which the caller
        can inspect afterwards: self.last_unreachable_recommendations (recommendations that
        could not be honored at all), self.last_runaway_cases (cases force-truncated by
        max_events_per_case), self.last_non_fitting_prefixes and
        self.last_model_inserted_activities (informational, from prefix reconstruction --
        see _reconstruct_prefix_state).

        Args:
            n_traces (int, optional): number of brand-new cases to generate. Ignored (forced
                to 0) when `prev_log` is given. Defaults to 1.
            t_start (datetime, optional): simulation start time for newly generated cases'
                arrivals. Ignored (overridden by the latest prefix start time in `prev_log`)
                when `prev_log` is given. Defaults to datetime.now().
            deterministic_time (bool, optional): if True, use each distribution's stored
                mean/representative value instead of sampling randomly -- for reproducible,
                "expected value" runs rather than stochastic ones. Defaults to False.
            prev_log (pd.DataFrame, optional): historical event log to continue from (see
                "Recommendation / prefix-continuation" above). Must contain, per case, rows
                with "case:concept:name", "org:resource", "start:timestamp",
                "time:timestamp", and (for the rows to actually continue) non-null
                "recommendation:act"/"recommendation:res" on the last row of each prefix to
                continue. Defaults to None (pure generation mode).
            case fires more than this many NEW (simulated, non-historical) visible events,
            it is force-truncated -- logged to self.last_runaway_cases -- instead of being
            left to spin (possibly forever) and starve every other case in the batch. This
            was observed to happen on BPI12: a case whose historical prefix does not perfectly
            fit the model (see self.last_non_fitting_prefixes) can be seeded with a marking
            that permanently disables the loop's only exit transition, turning a discovered
            model loop into a 100%-probability infinite cycle for that case. Set to None to
            disable the cap and restore the previous (uncapped) behaviour.

        max_reachability_search_nodes / max_reachability_search_seconds: safety valve for
            _bfs_path_to_activity, the search used to find a legal path to a recommended
            activity. A case seeded from a non-fitting historical prefix can carry a marking
            whose reachable state space explodes combinatorially (observed on BPI12: 70k+
            distinct markings visited, 10+ seconds, for a single case) -- this caps the
            search instead of letting it hang. When the cap is hit, the recommendation is
            reported as unreachable with a reason that makes clear it is undetermined (search
            aborted) rather than proven impossible -- see self.last_unreachable_recommendations.

        Returns:
            pd.DataFrame: the resulting event log, with columns "case:concept:name",
            "concept:name", "org:resource", "start:timestamp", "time:timestamp" (plus any
            case/event data-attribute columns), sorted by start/end timestamp. Contains only
            the newly simulated events in pure-generation mode, or `prev_log`'s rows
            concatenated with the simulated continuation events in prefix-continuation mode.
        """
        ##########################################
        #SETUP of the simulator
        ##########################################
        self.last_unreachable_recommendations = []
        self.last_runaway_cases = []
        self.last_non_fitting_prefixes = []
        self.last_model_inserted_activities = []
        self._reachability_search_max_nodes = max_reachability_search_nodes
        self._reachability_search_max_seconds = max_reachability_search_seconds

        event_log = []
        enabled_heap = [] # will have (time, case_id)
        resource_schedule = {r: [] for r in self.simulation_parameters.resources} # {resource: [(start, end), ...]}
        resource_starts = {r: [] for r in self.simulation_parameters.resources} # {resource: [start, ...]}
        resource_ends = {r: [] for r in self.simulation_parameters.resources} # {resource: [end, ...]}
        cases = []

        ##########################################
        #MANAGING the prev_log
        ##########################################
        if prev_log is not None: #only if we want to continue the prefix
            if prev_log.empty:
                raise ValueError(
                    "prev_log is empty. Verify case-id filtering and input log content before simulation."
                )
            if "recommendation:act" not in prev_log.columns:
                prev_log = prev_log.copy()
                prev_log["recommendation:act"] = None
            if "recommendation:res" not in prev_log.columns:
                if "recommendation:act" not in prev_log.columns:
                    prev_log = prev_log.copy()
                prev_log["recommendation:res"] = None
            trace_durations = prev_log.groupby("case:concept:name").agg(
                trace_start=('start:timestamp', 'min'),
                trace_end=('time:timestamp', 'max')
            )
            if trace_durations.empty:
                raise ValueError(
                    "No cases found in prev_log after grouping by case:concept:name."
                )
            # Only the historical prefixes themselves are simulated -- no additional new cases are sampled to start alongside them.
            n_traces = 0

            t_start = trace_durations["trace_start"].max() #time of the last event in the prefix

            # update the resource_schedule with the previous log
            for _, row in prev_log.iterrows(): 
                res = row['org:resource']
                if res not in resource_schedule:
                    resource_schedule[res] = []
                    resource_starts[res] = []
                    resource_ends[res] = []
                resource_schedule[res].append((row['start:timestamp'], row['time:timestamp']))
                bisect.insort(resource_starts[res], row['start:timestamp'])
                bisect.insort(resource_ends[res], row['time:timestamp'])

            # filter only the prefixes
            cases_prefixes = list(prev_log[~prev_log['recommendation:act'].isna() | ~prev_log['recommendation:res'].isna()]["case:concept:name"].unique())
            n_prefixes = len(cases_prefixes)
            prefixes_log = prev_log[prev_log['case:concept:name'].isin(cases_prefixes)]
        
        else:
            n_prefixes = 0

        ##########################################
        # SAMPLING
        ##########################################
        effective_n_traces = max(n_traces, 1)

        if not self.simulation_parameters.rules_mode: #if max_depth = 0
            if deterministic_time:
                sampled_arrivals = self.simulation_parameters.arrival_time_distribution[-1]
                sampled_waiting_times = {res : self.simulation_parameters.waiting_time_distributions[res][-1] for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: self.simulation_parameters.execution_time_distributions[act][-1] for act in self.simulation_parameters.net_transition_labels}
            else:
                sampled_arrivals = sampling_from_dist(*self.simulation_parameters.arrival_time_distribution, n_sample=effective_n_traces)
                sampled_waiting_times = {res : sampling_from_dist(*self.simulation_parameters.waiting_time_distributions[res], n_sample=effective_n_traces) for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: sampling_from_dist(*self.simulation_parameters.execution_time_distributions[act], n_sample=effective_n_traces) for act in self.simulation_parameters.net_transition_labels}

        if self.simulation_parameters.label_data_attributes: # when generating new cases we sample the case attributes
            x_attr_list = random.choices(
                list(self.simulation_parameters.distribution_data_attributes.keys()), 
                weights=list(self.simulation_parameters.distribution_data_attributes.values()),
                k = n_traces
                )
            x_attr_list = [list(attr) for attr in x_attr_list]
        else:
            x_attr_list = [[]]*n_traces

        ##########################################
        # INITIALIZE CASES
        ##########################################
        current_arr_ts = t_start

        if prev_log is not None: # only for continuation of prefixes
            rename_case_id = dict()
            for c in range(n_prefixes):
                # extract values of prefix c + recommendation 
                case_id_c = cases_prefixes[c] #id of prefix (case:concept:name)
                rename_case_id[f"case_{c+1}"] = case_id_c
                prefix_log_c = prefixes_log[prefixes_log['case:concept:name'] == case_id_c]
                prefix_log_c_sorted = prefix_log_c.sort_values('time:timestamp')
                prefix_end_c = prefix_log_c_sorted['time:timestamp'].iloc[-1]
                rec_act_c = prefix_log_c_sorted['recommendation:act'].iloc[-1]
                rec_res_c = prefix_log_c_sorted['recommendation:res'].iloc[-1]

                # reconstruct state of prefix c using alignment 
                prefix_state = self._reconstruct_prefix_state(case_id_c, prefix_log_c_sorted)
                current_marking_c = prefix_state["marking"]
                history_c = prefix_state["history"] #how many times each activity has been visited

                # if is_fit = false for prefix c we save the log moves and save the case_id 
                if not prefix_state["is_fit"]:
                    self.last_non_fitting_prefixes.append({
                        "case:concept:name": str(case_id_c),
                        "reason": ("alignment-based replay of the historical prefix needed at least one "
                                   "'log move' -- a logged activity the model could not explain at all "
                                   "(a genuine deviation). Informational only: unlike the old token-based "
                                   "replay, the reconstructed marking is always legally reachable, so this "
                                   "does not by itself put the case at risk of an unrecoverable loop."),
                    })
                if prefix_state["inserted_activities"]:
                    self.last_model_inserted_activities.append({
                        "case:concept:name": str(case_id_c),
                        "inserted_activities": ";".join(prefix_state["inserted_activities"]),
                        "reason": ("alignment needed to insert visible activity(ies) not present in the "
                                   "historical log to explain how the model reaches a state consistent "
                                   "with the logged prefix. Never written to the output log -- the saved "
                                   "prefix is always exactly the input log, unchanged -- but these ARE "
                                   "counted in this case's activity-history used for probabilistic "
                                   "weighting of subsequent simulated events."),
                    })

                # case attribute extraction from log
                trace_attributes = prefix_log_c[self.simulation_parameters.label_data_attributes].iloc[-1].to_dict()
                trace_attributes_c = dict()
                if self.simulation_parameters.label_data_attributes:
                    for  a in self.simulation_parameters.label_data_attributes:
                        if a in self.simulation_parameters.label_data_attributes_categorical: #one-hot encoding
                            for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                                trace_attributes_c[a+' = '+str(v)] = int(trace_attributes[a] == v)
                        else:
                            trace_attributes_c[a] = trace_attributes[a]

                #case creation
                case = {
                        "arrival_time": prefix_end_c,                   #end of the prefix
                        "case_id": c,                                   #index that represent the case id
                        "case_external_id": case_id_c,                  #real case id 
                        "marking": current_marking_c,                   #current marking at the end of the prefix
                        "place_token_time": {},
                        "enabled": {},
                        "history": history_c,                           #counts from alignment 
                        "attributes": trace_attributes_c,               #case attributes
                        "rec_act": rec_act_c,                           #recommendation activity
                        "rec_res": rec_res_c,                           #recommendation resource
                        "pending_invisible_path": [],                   #will contain the invisible transition used to get to the reccomendation if needed
                        "recommendation_lock_until": None,              #will be true after the recommendation will be executed
                        "strict_recommendation": pd.notna(rec_act_c),   #true only if the recommendation is something real, false for the baseline 
                        "sim_event_count": 0,                           #how many events are generated for this case
                    }

                #place token time is used to compute when a transition is enabled, we need to have all input's token to fire, the time will be the max between those
                for place in self.net.places:
                    case["place_token_time"][place] = None
                for place in current_marking_c.keys():
                    case["place_token_time"][place] = case["arrival_time"]
                enabled = self._get_enabled_transitions_sorted(case["marking"])
                #for each transition we understand when it becomes enabled 
                for t in enabled:
                    input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                    enabled_time = max(case["place_token_time"][p] for p in input_places)
                    enabled_time = max(enabled_time, prefix_end_c)
                    enabled_time = self._apply_recommendation_lock(case, enabled_time, t)
                    case["enabled"][t] = enabled_time

                #baseline sentinel
                is_sentinel_rec_act = isinstance(rec_act_c, str) and rec_act_c.startswith("__NO_RECOMMENDATION__")

                if case["enabled"]: 
                    #there is at least an enabled transition, we take the closest one in time and put it in the heap
                    enabled_time_case = min(case["enabled"].values())
                    heapq.heappush(enabled_heap, (enabled_time_case, c))
                elif case["strict_recommendation"] and not is_sentinel_rec_act:
                    # No transition is enabled at all right after replaying the historical  prefix, so this case would silently never be pushed onto the heap and the recommendation would never be attempted or reported. Two known causes: the replayed marking already equals the final marking (the "prefix" was actually the full, already-completed case, or the prefix replay left the case in a genuine deadlock (no legal continuation at all). Either way this is a "genuinely not possible" case per the strict-recommendation contract, so it belongs in last_unreachable_recommendations like any other failure.
                    reason = (
                        "case_already_complete_after_replayed_prefix"
                        if case["marking"] == self.final_marking
                        else "no_enabled_transitions_after_replayed_prefix_deadlock"
                    )
                    self.last_unreachable_recommendations.append(
                        {
                            "case:concept:name": str(case_id_c),
                            "recommendation:act": rec_act_c,
                            "recommendation:res": rec_res_c,
                            "reason": reason,
                        }
                    )

                cases.append(case)

        # when generating cases from scratch
        for i in range(n_traces):

            trace_attributes = dict()
            if x_attr_list[i]:
                for j, a in enumerate(self.simulation_parameters.label_data_attributes):
                    if a in self.simulation_parameters.label_data_attributes_categorical:
                        for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                            trace_attributes[a+' = '+str(v)] = int(x_attr_list[i][j] == v)
                    else:
                        trace_attributes[a] = x_attr_list[i][j]
                
            else:
                trace_attributes = dict()

            if not self.simulation_parameters.rules_mode:
                if deterministic_time:
                    arrival_delta = sampled_arrivals
                else:
                    arrival_delta = sampled_arrivals[i]
            else:
                if deterministic_time:
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.apply({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
                else:
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
            if arrival_delta == 0:
                arrival_delta = 1
            current_arr_ts = add_minutes_with_calendar(current_arr_ts, int(arrival_delta), self.simulation_parameters.arrival_calendar)

            case = {
                "case_id": i + n_prefixes,
                "case_external_id": f"case_{i + n_prefixes + 1}",
                "marking": self.initial_marking,
                "arrival_time": current_arr_ts,
                "place_token_time": {},
                "enabled": {},
                "history": {t: 0 for t in self.simulation_parameters.net_transition_labels},
                "attributes": trace_attributes,
                "rec_act": None,
                "rec_res": None,
                "pending_invisible_path": [],
                "recommendation_lock_until": None,
                "strict_recommendation": False,
                "sim_event_count": 0,
            }
            for place in self.net.places:
                case["place_token_time"][place] = None
            case["place_token_time"][list(self.initial_marking.keys())[0]] = case["arrival_time"]

            enabled = self._get_enabled_transitions_sorted(case["marking"])
            for t in enabled:
                input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                enabled_time = max(case["place_token_time"][p] for p in input_places)
                enabled_time = self._apply_recommendation_lock(case, enabled_time, t)
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                enabled_time_case = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (enabled_time_case, i + n_prefixes))

            cases.append(case)

        attribute_columns = list(cases[0]["attributes"].keys()) if cases and cases[0]["attributes"] else []

        ##########################################
        # START SIMULATION
        ##########################################
        completed_cases = set()
        if prev_log is not None:
            progress_total = n_prefixes + n_traces
        else:
            progress_total = n_traces
        pbar = tqdm(total=max(progress_total, 1), desc="Simulating Cases")
        while enabled_heap:
            _, case_id = heapq.heappop(enabled_heap) #pop the one with smallest time
            case = cases[case_id]

            if not case["enabled"]:
                continue

            enabled_transitions = self._sort_transitions(case["enabled"].keys())
            flag_rec = False
            chosen_transition, activity, t_enabled, flag_rec = self._resolve_recommended_transition(case, enabled_transitions) #look if there is recommendation, flag_reg true if the rec is possible 


            #flag_rec true if we have the rec (possible) or an invisible transition needed to get to the rec 
            if not flag_rec: #if the next activity is not the rec or the baseline 
                if not self.simulation_parameters.rules_mode: #if rules_mode false 
                    transition_weights = self.simulation_parameters.transition_weights
                else: #if rules mode true
                    transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, case["attributes"] | case["history"], enabled_transitions)
                chosen_transition = return_fired_transition(transition_weights, enabled_transitions) #given the transition available it extract the chosen one 
                activity = chosen_transition.label
                t_enabled = case["enabled"][chosen_transition]
            elif activity is not None: #rec executed
                case["rec_act"] = None #set to none so that the model understand that there are no more constraints

            if activity is not None: 
                #chosen transition correspond to a real one (no invisible transition)

                #resource 
                if flag_rec and case["rec_res"] is not None:
                    #rec and resource to use -> we force it
                    resource = case["rec_res"]
                    t_enabled_waited = t_enabled
                    r_workload = count_concurrent_events_fast(
                        resource_starts.get(resource, []), resource_ends.get(resource, []), t_enabled
                    )
                else:
                    # choice of the resource based on the workload at time t_enabled
                    workloads = {
                        r: count_concurrent_events_fast(resource_starts[r], resource_ends[r], t_enabled)
                        for r in self.simulation_parameters.resources
                    }
                    enabled_resources_act = [r for r, v in self.simulation_parameters.act_resource_prob[activity].items() if v>0]
                    enabled_resources = []
                    for r in enabled_resources_act:
                        #keep only the available one between the qualified ones
                        if workloads[r] == 0:
                            enabled_resources.append(r)
                        else:
                            if r in self.simulation_parameters.multitasking_resources:
                                enabled_resources.append(r)

                    if not enabled_resources:
                        # if there are no available resources we chose the one that becomes free first, so we have a waiting time for the resource
                        t_enabled_enabled_resources = [resource_schedule[r][-1][-1] for r in enabled_resources_act]
                        index_res, t_enabled_waited = min(enumerate(t_enabled_enabled_resources), key=lambda x: x[1])
                        resource = enabled_resources_act[index_res]
                    else:
                        # if there are enabled resources we extract it, no waiting time 
                        resource_weights = self.simulation_parameters.act_resource_prob[activity]
                        resource = return_resource(resource_weights, enabled_resources)
                        t_enabled_waited = t_enabled
                    r_workload = workloads[resource]

                #waiting time
                if sum(case["history"].values()) == 0: #it the activity if the first one 
                    waiting_time = 0
                else:
                    if not self.simulation_parameters.rules_mode:
                        if deterministic_time:
                            waiting_time = sampled_waiting_times[resource]
                            try:
                                int(waiting_time)
                            except:
                                waiting_time = 0
                        else:
                            candidate_waiting_times = list(sampled_waiting_times[resource])
                            if candidate_waiting_times:
                                waiting_time = random.choice(candidate_waiting_times)
                            else:
                                waiting_time = 0
                    else: #rules_mode = TRUE
                        if deterministic_time:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply({'workload': r_workload} | case["history"] | case["attributes"])
                            try:
                                int(waiting_time)
                            except:
                                waiting_time = 0
                        else: #sample waiting time 
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution({'workload': r_workload} | case["history"] | case["attributes"])

                #t_enabled_waited - t_enabled = time used for the resource to free itself, we subtract in order to not count it twice
                waiting_time -= (t_enabled_waited - t_enabled).total_seconds() / 60
                waiting_time = max(0, waiting_time)
                t_start_exec = add_minutes_with_calendar(t_enabled_waited, int(waiting_time), self.simulation_parameters.calendars[resource]) #time when the next activity starts but it must respect the resource calendar (otherwise it will be postponed)

                #execution time
                if not self.simulation_parameters.rules_mode:
                    if deterministic_time:
                        ex_time = sampled_execution_times[activity]
                        try:
                            int(ex_time)
                        except:
                            ex_time = 0
                    else: 
                        candidate_execution_times = list(sampled_execution_times[activity])
                        if candidate_execution_times:
                            ex_time = random.choice(candidate_execution_times)
                        else:
                            ex_time = 0
                else: #rules_mode TRUE
                    if deterministic_time:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])
                        try:
                            int(ex_time)
                        except: 
                            ex_time = 0
                    else: #extraction from trees
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])

                # end time for the execution of the activity considering the resource calendar
                t_end = add_minutes_with_calendar(t_start_exec, int(ex_time), self.simulation_parameters.calendars[resource])

                # event added to the log 
                event_log.append((case_id, activity, resource, t_enabled, t_start_exec, t_end) + tuple(case['attributes'].values()))
                if resource not in resource_schedule:
                    resource_schedule[resource] = []
                    resource_starts[resource] = []
                    resource_ends[resource] = []
                resource_schedule[resource].append((t_start_exec, t_end))
                bisect.insort(resource_starts[resource], t_start_exec)
                bisect.insort(resource_ends[resource], t_end)
                case["history"][activity] += 1
                case["sim_event_count"] = case.get("sim_event_count", 0) + 1

                #check if the rec is the chosen one we lock and fix that no other visible event can start before this moment
                if flag_rec and chosen_transition is not None and chosen_transition.label == activity:
                    case["recommendation_lock_until"] = t_end
            else: # invisible transition
                t_end = t_enabled

            if chosen_transition is None:
                case["enabled"] = {}
                enabled = self._get_enabled_transitions_sorted(case["marking"])
                for t in enabled:
                    input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                    enabled_time = max(case["place_token_time"][p] for p in input_places)
                    enabled_time = self._apply_recommendation_lock(case, enabled_time, t)
                    case["enabled"][t] = enabled_time

                if case["enabled"]:
                    next_enabled_time = min(case["enabled"].values())
                    heapq.heappush(enabled_heap, (next_enabled_time, case_id))
                continue

            #marking update
            for arc in chosen_transition.out_arcs:
                case["place_token_time"][arc.target] = t_end
            case["enabled"] = {}
            can_fire = True
            for arc in chosen_transition.in_arcs:
                if case["marking"].get(arc.source, 0) < arc.weight:
                    can_fire = False
                    break
            if can_fire:
                case["marking"] = update_current_marking(case["marking"], chosen_transition)
            #if we are at the final marking we stop and go to the next event
            if case["marking"] == self.final_marking:
                if case_id not in completed_cases:
                    pbar.update(1)
                    completed_cases.add(case_id)
                continue

            #if the case generate too many events it will be cut
            if max_events_per_case is not None and case.get("sim_event_count", 0) >= max_events_per_case:
                self.last_runaway_cases.append({
                    "case:concept:name": str(case.get("case_external_id", case.get("case_id"))),
                    "sim_events_generated": case.get("sim_event_count", 0),
                    "reason": (f"case fired >= max_events_per_case={max_events_per_case} simulated events "
                               "without reaching the final marking and was truncated to avoid a "
                               "runaway/near-infinite loop; see last_non_fitting_prefixes for the most "
                               "common root cause on prefix-continuation runs."),
                })
                if case_id not in completed_cases:
                    pbar.update(1)
                    completed_cases.add(case_id)
                continue

            # if case non completed we compute the new transition and put the case back in the heap
            enabled = self._get_enabled_transitions_sorted(case["marking"])
            for t in enabled:
                input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                enabled_time = max(case["place_token_time"][p] for p in input_places)
                enabled_time = self._apply_recommendation_lock(case, enabled_time, t)
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                next_enabled_time = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (next_enabled_time, case_id))

        pbar.close()

        ##########################################
        # Final Dataframe Construction
        ########################################## 
        df_log = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name", "org:resource", "enabled:timestamp", "start:timestamp", "time:timestamp"] + attribute_columns)
        df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: f"case_{x+1}")
        if prev_log is not None:
            df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: rename_case_id.get(x, x))
            df_log = pd.concat([prev_log, df_log], ignore_index=True)
            df_log = df_log[["case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"] + attribute_columns]
        df_log.sort_values(by=["start:timestamp", "time:timestamp"], inplace=True)
        df_log.reset_index(drop=True, inplace=True)

        return df_log