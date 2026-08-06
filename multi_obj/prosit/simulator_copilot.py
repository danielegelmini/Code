import random
import pandas as pd
import math
import heapq
from datetime import datetime
from tqdm import tqdm

from typing import Union, List, Dict, Optional

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

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
        """ Initilize parameters """
        
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
        if isinstance(key, tuple):
            return f"__tuple__:{json.dumps(key)}"
        if isinstance(key, (str, int, float, bool)) or key is None:
            return key
        return str(key)

    @staticmethod
    def _restore_from_json(value):
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
        """ Discovery Parameters from event log data """

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

        dict_params = self.to_dict()
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump(dict_params, json_file, indent=4)
        os.replace(temp_path, path)


    def from_dict(self, dict_params):

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

        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters
        self._transition_rank = self._build_transition_rank()
        self.raise_on_unreachable_recommendation = False
        self.last_unreachable_recommendations = []

    def _build_transition_rank(self) -> dict:
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
        return sorted(
            list(transitions),
            key=lambda t: (
                self._transition_rank.get(t, math.inf),
                str(t.label) if t.label is not None else "",
                str(getattr(t, "name", "")),
            ),
        )

    def _get_enabled_transitions_sorted(self, marking) -> list:
        return self._sort_transitions(return_enabled_transitions(self.net, marking))

    def _get_case_enabled_time(self, case) -> datetime:
        if case["enabled"]:
            return min(case["enabled"].values())
        return case["arrival_time"]

    def _find_invisible_path_to_activity(self, case, target_activity, max_depth=200):
        from collections import deque

        start_marking = case["marking"]
        queue = deque([(start_marking, [])])
        visited = {frozenset(start_marking.items())}

        while queue:
            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            enabled = self._get_enabled_transitions_sorted(marking)
            if any(t.label == target_activity for t in enabled):
                return path

            for t in enabled:
                if t.label is not None:
                    continue
                next_marking = update_current_marking(marking, t)
                key = frozenset(next_marking.items())
                if key not in visited:
                    visited.add(key)
                    queue.append((next_marking, path + [t]))

        return None

    def _find_path_to_activity(self, case, target_activity, max_depth=200):
        from collections import deque

        start_marking = case["marking"]
        queue = deque([(start_marking, [])])
        visited = {frozenset(start_marking.items())}

        while queue:
            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            enabled = self._get_enabled_transitions_sorted(marking)

            # Stop as soon as target is directly enabled from this marking.
            if any(t.label == target_activity for t in enabled):
                return path

            for t in enabled:
                next_marking = update_current_marking(marking, t)
                key = frozenset(next_marking.items())
                if key not in visited:
                    visited.add(key)
                    queue.append((next_marking, path + [t]))

        return None

    def _resolve_recommended_transition(self, case, enabled_transitions):
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

        invisible_path = self._find_invisible_path_to_activity(case, case["rec_act"])
        if invisible_path:
            chosen_transition = invisible_path[0]
            case["pending_invisible_path"] = invisible_path[1:]
            return chosen_transition, None, case["enabled"].get(chosen_transition, self._get_case_enabled_time(case)), True

        # If invisible-only path does not exist, walk the deterministic shortest
        # path until recommendation becomes enabled while still emitting the
        # recommendation as the first post-prefix visible activity.
        reach_path = self._find_path_to_activity(case, case["rec_act"])
        if reach_path:
            chosen_transition = reach_path[0]
            case["pending_invisible_path"] = reach_path[1:]
            return chosen_transition, None, case["enabled"].get(chosen_transition, self._get_case_enabled_time(case)), True

        # In strict mode, never consume other visible activities before recommendation.
        if case.get("strict_recommendation", False):
            case_external_id = case.get("case_external_id", case.get("case_id"))
            rec_act = case.get("rec_act")
            rec_res = case.get("rec_res")
            self.last_unreachable_recommendations.append(
                {
                    "case:concept:name": str(case_external_id),
                    "recommendation:act": rec_act,
                    "recommendation:res": rec_res,
                    "reason": "not_reachable_from_replayed_prefix_marking",
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
        generate_new_cases_from_prev_log: bool = True,
    ) -> pd.DataFrame:

        self.last_unreachable_recommendations = []

        event_log = []
        enabled_heap = []
        resource_schedule = {r: [] for r in self.simulation_parameters.resources}
        cases = []

        if prev_log is not None:
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

            # computing the number of traces we want to simulate as the number of traces that started during the longest trace in the previous log
            trace_durations = prev_log.groupby("case:concept:name").agg(
                trace_start=('start:timestamp', 'min'),
                trace_end=('time:timestamp', 'max')
            )
            if trace_durations.empty:
                raise ValueError(
                    "No cases found in prev_log after grouping by case:concept:name."
                )
            trace_durations['duration'] = trace_durations['trace_end'] - trace_durations['trace_start']
            longest_trace = trace_durations.loc[trace_durations['duration'].idxmax()]
            longest_start = longest_trace['trace_start']
            longest_end = longest_trace['trace_end']
            longest_case = trace_durations['duration'].idxmax()
            started_during_longest = trace_durations[
                (trace_durations['trace_start'] >= longest_start) &
                (trace_durations['trace_start'] <= longest_end) &
                (trace_durations.index != longest_case)
            ]
            if generate_new_cases_from_prev_log:
                n_traces = started_during_longest.shape[0]
            else:
                n_traces = 0

            # compute the t_start
            t_start = trace_durations["trace_start"].max()

            # update the resource_schedule with the previous log
            for _, row in prev_log.iterrows():
                if row['org:resource'] not in resource_schedule:
                    resource_schedule[row['org:resource']] = []
                resource_schedule[row['org:resource']].append((row['start:timestamp'], row['time:timestamp']))

            # filter only the prefixes
            cases_prefixes = list(prev_log[~prev_log['recommendation:act'].isna() | ~prev_log['recommendation:res'].isna()]["case:concept:name"].unique())
            n_prefixes = len(cases_prefixes)
            prefixes_log = prev_log[prev_log['case:concept:name'].isin(cases_prefixes)]
        
        else:
            n_prefixes = 0

        effective_n_traces = max(n_traces, 1)

        if not self.simulation_parameters.rules_mode:
            if deterministic_time:
                sampled_arrivals = self.simulation_parameters.arrival_time_distribution[-1]
                sampled_waiting_times = {res : self.simulation_parameters.waiting_time_distributions[res][-1] for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: self.simulation_parameters.execution_time_distributions[act][-1] for act in self.simulation_parameters.net_transition_labels}
            else:
                sampled_arrivals = sampling_from_dist(*self.simulation_parameters.arrival_time_distribution, n_sample=effective_n_traces)
                sampled_waiting_times = {res : sampling_from_dist(*self.simulation_parameters.waiting_time_distributions[res], n_sample=effective_n_traces) for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: sampling_from_dist(*self.simulation_parameters.execution_time_distributions[act], n_sample=effective_n_traces) for act in self.simulation_parameters.net_transition_labels}

        if self.simulation_parameters.label_data_attributes:
            x_attr_list = random.choices(
                list(self.simulation_parameters.distribution_data_attributes.keys()), 
                weights=list(self.simulation_parameters.distribution_data_attributes.values()),
                k = n_traces
                )
            x_attr_list = [list(attr) for attr in x_attr_list]
        else:
            x_attr_list = [[]]*n_traces

        current_arr_ts = t_start

        # INITIALIZE CASES

        if prev_log is not None:
            rename_case_id = dict()
            for c in range(n_prefixes):
                case_id_c = cases_prefixes[c]
                rename_case_id[f"case_{c+1}"] = case_id_c
                prefix_log_c = prefixes_log[prefixes_log['case:concept:name'] == case_id_c]
                prefix_log_c_sorted = prefix_log_c.sort_values('time:timestamp')
                prefix_end_c = prefix_log_c_sorted['time:timestamp'].iloc[-1]
                rec_act_c = prefix_log_c_sorted['recommendation:act'].iloc[-1]
                rec_res_c = prefix_log_c_sorted['recommendation:res'].iloc[-1]

                replayed = token_replay.apply(
                                                prefix_log_c_sorted,
                                                self.simulation_parameters.net, 
                                                self.simulation_parameters.initial_marking, 
                                                self.simulation_parameters.final_marking
                                            )

                current_marking_c = replayed[0]["reached_marking"]

                history_c_list = [t_l.label for t_l in replayed[0]["activated_transitions"] if t_l.label]
                history_c = {t: 0 for t in self.simulation_parameters.net_transition_labels}
                for t in history_c_list:
                    history_c[t] += 1

                trace_attributes = prefix_log_c[self.simulation_parameters.label_data_attributes].iloc[-1].to_dict()
                trace_attributes_c = dict()
                if self.simulation_parameters.label_data_attributes:
                    for  a in self.simulation_parameters.label_data_attributes:
                        if a in self.simulation_parameters.label_data_attributes_categorical:
                            for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                                trace_attributes_c[a+' = '+str(v)] = int(trace_attributes[a] == v)
                        else:
                            trace_attributes_c[a] = trace_attributes[a]

                case = {
                    "arrival_time": prefix_end_c,
                        "case_id": c,
                        "case_external_id": case_id_c,
                        "marking": current_marking_c,
                        "place_token_time": {},
                        "enabled": {},
                        "history": history_c,
                        "attributes": trace_attributes_c,
                        "rec_act": rec_act_c,
                        "rec_res": rec_res_c,
                        "pending_invisible_path": [],
                        "recommendation_lock_until": None,
                        "strict_recommendation": pd.notna(rec_act_c),
                    }

                for place in self.net.places:
                    case["place_token_time"][place] = None
                for place in current_marking_c.keys():
                    case["place_token_time"][place] = case["arrival_time"]

                enabled = self._get_enabled_transitions_sorted(case["marking"])
                for t in enabled:
                    input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                    enabled_time = max(case["place_token_time"][p] for p in input_places)
                    enabled_time = max(enabled_time, prefix_end_c)
                    enabled_time = self._apply_recommendation_lock(case, enabled_time, t)
                    case["enabled"][t] = enabled_time

                if case["enabled"]:
                    enabled_time_case = min(case["enabled"].values())
                    heapq.heappush(enabled_heap, (enabled_time_case, c))

                cases.append(case)

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

        # START SIMULATION
        completed_cases = set()
        if prev_log is not None:
            progress_total = n_prefixes + n_traces
        else:
            progress_total = n_traces
        pbar = tqdm(total=max(progress_total, 1), desc="Simulating Cases")
        while enabled_heap:
            _, case_id = heapq.heappop(enabled_heap)
            case = cases[case_id]

            if not case["enabled"]:
                continue

            enabled_transitions = self._sort_transitions(case["enabled"].keys())
            flag_rec = False
            chosen_transition, activity, t_enabled, flag_rec = self._resolve_recommended_transition(case, enabled_transitions)

            if not flag_rec:
                if not self.simulation_parameters.rules_mode:
                    transition_weights = self.simulation_parameters.transition_weights
                else:
                    transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, case["attributes"] | case["history"])
                chosen_transition = return_fired_transition(transition_weights, enabled_transitions)
                activity = chosen_transition.label
                t_enabled = case["enabled"][chosen_transition]
            elif activity is not None:
                case["rec_act"] = None

            if activity is not None:
                if flag_rec and case["rec_res"] is not None:
                    resource = case["rec_res"]
                    t_enabled_waited = t_enabled
                    r_workload = count_concurrent_events(resource_schedule[resource], t_enabled)
                else:
                    workloads = {r: count_concurrent_events(resource_schedule[r], t_enabled) for r in self.simulation_parameters.resources}
                    enabled_resources_act = [r for r, v in self.simulation_parameters.act_resource_prob[activity].items() if v>0]
                    enabled_resources = []
                    for r in enabled_resources_act:
                        if workloads[r] == 0:
                            enabled_resources.append(r)
                        else:
                            if r in self.simulation_parameters.multitasking_resources:
                                enabled_resources.append(r)
                    if not enabled_resources:
                        t_enabled_enabled_resources = [resource_schedule[r][-1][-1] for r in enabled_resources_act]
                        index_res, t_enabled_waited = min(enumerate(t_enabled_enabled_resources), key=lambda x: x[1])
                        resource = enabled_resources_act[index_res]
                    else:
                        resource_weights = self.simulation_parameters.act_resource_prob[activity]
                        resource = return_resource(resource_weights, enabled_resources)
                        t_enabled_waited = t_enabled
                    r_workload = workloads[resource]
                
                if sum(case["history"].values()) == 0:
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
                    else:
                        if deterministic_time:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply({'workload': r_workload} | case["history"] | case["attributes"])
                            try:
                                int(waiting_time)
                            except:
                                waiting_time = 0
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution({'workload': r_workload} | case["history"] | case["attributes"])

                waiting_time -= (t_enabled_waited - t_enabled).total_seconds() / 60
                waiting_time = max(0, waiting_time)
                t_start_exec = add_minutes_with_calendar(t_enabled_waited, int(waiting_time), self.simulation_parameters.calendars[resource])

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
                else:    
                    if deterministic_time:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])
                        try:
                            int(ex_time)
                        except: 
                            ex_time = 0
                    else:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])

                
                t_end = add_minutes_with_calendar(t_start_exec, int(ex_time), self.simulation_parameters.calendars[resource])

                event_log.append((case_id, activity, resource, t_enabled, t_start_exec, t_end) + tuple(case['attributes'].values()))
                resource_schedule[resource].append((t_start_exec, t_end))
                case["history"][activity] += 1

                if flag_rec and chosen_transition is not None and chosen_transition.label == activity:
                    case["recommendation_lock_until"] = t_end
            else:
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
            if case["marking"] == self.final_marking:
                if case_id not in completed_cases:
                    pbar.update(1)
                    completed_cases.add(case_id)
                continue
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
        df_log = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name", "org:resource", "enabled:timestamp", "start:timestamp", "time:timestamp"] + attribute_columns)
        df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: f"case_{x+1}")
        if prev_log is not None:
            df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: rename_case_id.get(x, x))
            df_log = pd.concat([prev_log, df_log], ignore_index=True)
            df_log = df_log[["case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"] + attribute_columns]
        df_log.sort_values(by=["start:timestamp", "time:timestamp"], inplace=True)
        df_log.reset_index(drop=True, inplace=True)

        return df_log