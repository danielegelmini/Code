import random
import pandas as pd
from datetime import datetime, timedelta
from copy import copy

import pm4py
from tqdm import tqdm
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from prosit.utils.rule_utils import DecisionRules

from river.tree.hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier



def return_transitions_frequency(
        log: EventLog, 
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking
    ) -> dict:

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]

    frequency_t = {t: 0 for t in net.transitions}
    list_transitions = list(net.transitions)
    for trace in aligned_traces:
        for align in trace:
            name_t = align[1]
            for t in list_transitions:
                if t.name == name_t:
                    frequency_t[t] += 1
                    break

    return frequency_t


def return_enabled_and_fired_transitions(
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking, 
        trace_aligned: list
    ) -> tuple:

    visited_transitions = []
    is_fired = []
    tkns = list(initial_marking)
    enabled_transitions = return_enabled_transitions(net, tkns)
    for t_fired_name in trace_aligned:
        for t in net.transitions:
            if t.name == t_fired_name[1]:
                t_fired = t
                break
        not_fired_transitions = list(enabled_transitions-{t_fired})
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)
        visited_transitions.append(t_fired)
        is_fired.append(1)
        tkns = update_current_marking(tkns, t_fired)
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired


def update_current_marking(m: Marking, t_fired: PetriNet.Transition) -> Marking:

    m_out = copy(m)
    for a in t_fired.in_arcs:
        m_out[a.source] -= a.weight
        if m_out[a.source] == 0:
            del m_out[a.source]

    for a in t_fired.out_arcs:
        m_out[a.target] += a.weight

    return m_out


def return_enabled_transitions(net: PetriNet, tkns: Marking) -> set:
    
    enabled_t = set()
    list_transitions = list(net.transitions)
    for t in list_transitions:
        can_fire = True
        for a in t.in_arcs:
            required_tokens = getattr(a, "weight", 1) or 1
            available_tokens = tkns.get(a.source, 0) if hasattr(tkns, "get") else 0
            if available_tokens < required_tokens:
                can_fire = False
                break
        if can_fire:
            enabled_t.add(t)
    
    return enabled_t


def return_fired_transition(transition_weights: dict, enabled_transitions: list) -> PetriNet.Transition:

    if not enabled_transitions:
        return None

    weights = []
    for transition in enabled_transitions:
        weight = transition_weights.get(transition, 1)
        if weight is None:
            weight = 1
        weights.append(float(weight))

    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(list(enabled_transitions))

    random_value = random.uniform(0, total_weight)

    cumulative_weight = 0
    for transition, weight in zip(enabled_transitions, weights):
        cumulative_weight += weight
        if random_value <= cumulative_weight:
            return transition

    return list(enabled_transitions)[-1]
        

def compute_transition_weights_from_model(models_t: dict, dict_x: dict) -> dict:
    transition_weights = dict()
    list_transitions = list(models_t.keys())
    for t in list_transitions:
        if type(models_t[t]) in [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
            X = pd.DataFrame({k: [dict_x[k]] for k in dict_x.keys()})
            transition_weights[t] = compute_proba(models_t, t, X)
        elif type(models_t[t]) == HoeffdingAdaptiveTreeClassifier:
            try:
                transition_weights[t] = models_t[t].predict_proba_one(dict_x)[1]
            except:
                transition_weights[t] = 0
        elif type(models_t[t]) == DecisionRules:
            transition_weights[t] = models_t[t].apply(dict_x)
        elif type(models_t[t]) == float:
            transition_weights[t] = models_t[t] 
        else:
            transition_weights[t] = 1
    return transition_weights


def return_resource(resource_weights: dict, enabled_resources: list) -> str:

    total_weight = sum(resource_weights[s] for s in enabled_resources)
    random_value = random.uniform(0, total_weight)
    
    cumulative_weight = 0
    for s in enabled_resources:
        cumulative_weight += resource_weights[s]
        if random_value <= cumulative_weight:
            return s


def compute_proba(models_t: dict, t: PetriNet.Transition, X:pd.DataFrame) -> float:
    
    clf_t = models_t[t]
    
    return clf_t.predict_proba(X)[0,1]


def count_concurrent_events(schedule, t_enabled) -> int:

    count = 0
    for start, end in reversed(schedule):
        if end <= t_enabled:
            break
        if start <= t_enabled < end:
            count += 1
            
    return count


def count_false_hours(calendar: dict, start_ts: datetime, end_ts: datetime) -> int:
    false_hours_count = 0
    current_time = start_ts
    
    while current_time < end_ts:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour) == False:
            false_hours_count += 1
            
        current_time += timedelta(hours=1)

    return false_hours_count


def count_concurrent_events(schedule, t_enabled) -> int:
    # A list comprehension evaluates in C-speed and correctly checks all
    # overlapping timeframes without failing on unsorted end-times.
    return sum(1 for start, end in schedule if start <= t_enabled < end)


def count_concurrent_events_fast(starts_sorted: list, ends_sorted: list, t_enabled) -> int:
    """
    Equivalent to count_concurrent_events(list(zip(starts_sorted, ends_sorted)), t_enabled)
    (i.e. counts intervals with start <= t_enabled < end), but runs in O(log n) instead of
    O(n) by binary-searching two separately-maintained sorted lists of start/end timestamps,
    instead of linearly rescanning the whole resource history on every call.

    Because start <= end always holds for a valid (start, end) interval, "end <= t_enabled"
    implies "start <= t_enabled", so the count of currently-open intervals at t_enabled is
    simply (# intervals with start <= t_enabled) - (# intervals with end <= t_enabled).

    starts_sorted / ends_sorted must be kept sorted (e.g. via bisect.insort on every insert);
    they are unrelated to each other's ordering (an interval's start and end need not be at
    the same index in the two lists).
    """
    import bisect
    started = bisect.bisect_right(starts_sorted, t_enabled)
    ended = bisect.bisect_right(ends_sorted, t_enabled)
    return started - ended


def add_minutes_with_calendar(start_ts: datetime, minutes_to_add: int, calendar: dict) -> datetime:
    # 1. Safety Check: Verify the resource actually has working hours configured
    has_working_hours = any(
        work_status 
        for day_schedule in calendar.values() if isinstance(day_schedule, dict) 
        for work_status in day_schedule.values()
    )
    
    if not has_working_hours and minutes_to_add > 0:
        # Fallback: If the calendar is entirely empty, add time linearly to prevent an infinite freeze
        return start_ts + timedelta(minutes=minutes_to_add)
        
    remaining_minutes = float(minutes_to_add) 
    current_time = start_ts

    while remaining_minutes > 0:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour, False):
            minutes_in_current_hour = min(remaining_minutes, 60 - current_time.minute)
            current_time += timedelta(minutes=minutes_in_current_hour)
            remaining_minutes -= minutes_in_current_hour
        else:
            # Jump directly to the top of the next hour safely
            current_time = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    return current_time


def get_transition_from_name(t_fired_name: str, net: PetriNet) -> PetriNet.Transition:
    for t in net.transitions:
        if t.name == t_fired_name:
            return t
        

def build_df_features(log, net, im, fm, act_to_resources_prob, net_transition_labels, resources, label_data_attributes=[]):

    df_log = pm4py.convert_to_dataframe(log)
    df_log["start:timestamp"] = df_log["start:timestamp"].apply(lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp())
    df_log["time:timestamp"] = df_log["time:timestamp"].apply(lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp())


    aligned_traces = alignments.apply_log(log, net, im, fm, parameters={"ret_tuple_as_trans_desc": True})

    act_to_resources = {act: [r for r, v in act_to_resources_prob[act].items() if v>0] for act in net_transition_labels}

    dataset = []
    for i, trace in enumerate(tqdm(log)):

        trace_aligned = aligned_traces[i]["alignment"]

        case_id = trace[0]["case:concept:name"]
        history = {t_l: 0 for t_l in net_transition_labels}
        if label_data_attributes:
            try:
                trace_attributes = [trace[a] for a in label_data_attributes]
            except:
                trace_attributes = [trace[0][a] for a in label_data_attributes]
        else:
            trace_attributes = []

        marking = im
        j = 0
        transition_enabled_times = dict()
        current_t = trace[0]["start:timestamp"]
        for step in trace_aligned:
            if step[0][1] == ">>": # log move
                continue

            transition = get_transition_from_name(step[0][1], net)
            transition_label = transition.label


            prev_enabled_transitions = return_enabled_transitions(net, marking)

            for enabled in prev_enabled_transitions:
                if enabled not in transition_enabled_times:
                    transition_enabled_times[enabled] = current_t

            if step[1][0] == step[1][1]: # sync move
                resource = trace[j]["org:resource"]
                start_t = trace[j]["start:timestamp"]
                end_t = trace[j]["time:timestamp"]
                enabled_t = transition_enabled_times[transition]
                current_t = end_t
                df_log_filtered = df_log[(df_log["time:timestamp"] > enabled_t.timestamp()) & (df_log["start:timestamp"] < enabled_t.timestamp())]
                res_workload = (df_log_filtered['org:resource']==resource).sum()
                prev_enabled_resources = act_to_resources[transition_label]
                j += 1
            else: # model move
                resource = None
                enabled_t = None
                start_t = None
                end_t = None
                res_workload = None
                prev_enabled_resources = None

            del transition_enabled_times[transition]
            
            marking = update_current_marking(marking, transition)

            dataset.append((case_id, transition, transition_label, resource, enabled_t, start_t, end_t, prev_enabled_transitions, prev_enabled_resources, res_workload) + tuple(trace_attributes) + tuple(history.values()))
            
            if transition_label:
                history[transition_label] += 1

    df = pd.DataFrame(dataset, columns=["case_id", "transition", "transition_label", "resource", "enabled_t", "start_t", "end_t", "prev_enabled_transitions", "prev_enabled_resources", "res_workload"] + label_data_attributes + net_transition_labels)

    return df


def explore_invisible_path_to_activity(net, current_marking, target_activity, max_depth=10):
    """
    Use BFS to find the shortest path through invisible transitions to reach a target activity.
    
    Args:
        net: The Petri net
        current_marking: Current marking of the case
        target_activity: The label of the activity we want to reach
        max_depth: Maximum depth for searching (to avoid infinite loops)
    
    Returns:
        List of invisible transitions to fire in order, or None if no path found
    """
    from collections import deque
    
    # BFS to find shortest path through invisible transitions
    queue = deque([(current_marking, [])])
    visited = set()
    visited.add(frozenset(current_marking.items()))
    
    while queue:
        marking, path = queue.popleft()
        
        # Check if we've exceeded max depth
        if len(path) >= max_depth:
            continue
        
        # Get enabled transitions from this marking
        enabled = return_enabled_transitions(net, marking)
        
        # Check if target activity is now directly enabled
        for t in enabled:
            if t.label == target_activity:
                return path  # Found a path!
        
        # Explore only invisible transitions
        for t in enabled:
            if t.label is None:  # Invisible transition
                # Simulate firing this transition
                new_marking = update_current_marking(marking, t)
                marking_key = frozenset(new_marking.items())
                
                if marking_key not in visited:
                    visited.add(marking_key)
                    queue.append((new_marking, path + [t]))
    
    return None  # No path found
