"""
Step 1 of 0_estimate_start_times_vinci.py: discovery of the Prosimos simulation model, i.e. a BPMN file
(the control flow) and a JSON file (the simulation parameters).

The repository of Vinci et al. takes these two files ready-made; the recipe used here is the one of
Fracca et al. (CAiSE 2022, Sec. 5), who discover the parameters with pm4py, written in the JSON format of
Prosimos. Only the completion times of the log are used (the start times are what we want to estimate).
The four parameters:

  1. resource pools   -- the roles of Burattin et al. (2013), pm4py's organizational-roles discovery;
  2. calendars        -- per pool, the weekdays and hours in which its resources complete most events;
  3. inter-arrival    -- the best-fitting distribution of the time between consecutive case arrivals;
  4. branching        -- for every XOR split of the BPMN, the probability of each outgoing flow.

The activity durations are not discovered here: the estimator overwrites them at every iteration.

`log` is always a DataFrame with one row per event and the columns CASE (case id), ACT (activity),
RES (resource), END (completion time), sorted per case by completion time.
"""
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py

from .estimator import ACT, CASE, END, RES, find_best_fit_distribution

DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]  # pandas: Monday = 0
CASE_START, CASE_END = "__start__", "__end__"  # pseudo-activities before the first / after the last event


# =====================================================================================================
# The discovery
# =====================================================================================================
def discover_simulation_model(log, pnml_path, out_dir, calendar_coverage, seed):
    """Petri net -> BPMN, then the four simulation parameters from `log` (completion events), then the JSON.
    Returns (bpmn_path, json_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = out_dir / "simulation_model.bpmn"
    json_path = out_dir / "simulation_parameters.json"

    # BPMN: Prosimos reads BPMN, not Petri nets (no layout: Prosimos does not need it and pm4py's layouter
    # needs Graphviz). The element ids are random, and the JSON refers to them: both are rebuilt together.
    pm4py.write_bpmn(pm4py.convert_to_bpmn(*pm4py.read_pnml(str(pnml_path))), str(bpmn_path), auto_layout=False)
    bpmn = read_bpmn(bpmn_path)  # tasks of each activity, XOR splits and their outgoing flows

    pools = discover_resource_pools(log)                                  # 1. who does what
    calendars = discover_calendars(log, pools, calendar_coverage)         # 2. when they work
    arrivals = discover_inter_arrival_time(log, calendar_coverage, seed)  # 3. when cases arrive
    branching = discover_branching_probabilities(log, bpmn)               # 4. which path a case takes

    write_simulation_parameters(json_path, bpmn, pools, calendars, arrivals, branching)
    return bpmn_path, json_path


# =====================================================================================================
# 1. Resource pools
# =====================================================================================================
def discover_resource_pools(log):
    """Roles of Burattin et al. (pm4py, default threshold 0.65): starting from one role per activity (with
    the multiset of resources that executed it), the two most similar roles are merged while
        sim(R1, R2) = sum_r min(p1(r), p2(r)) / sum_r max(p1(r), p2(r)) > 0.65,
    p(r) = share of the role's executions done by resource r.
    Returns [{"id", "activities", "members", "all_resources"}]: "all_resources" are the resources that
    executed the role's activities, and can execute them in the simulation; "members" only those whose main
    role it is (most executions), because in Prosimos a resource has a single pool and calendar."""
    # Each role: .activities = list of activities, .originator_importance = {resource: n. of executions}
    roles = pm4py.discover_organizational_roles(log, activity_key=ACT, resource_key=RES,
                                                timestamp_key=END, case_id_key=CASE)
    work = {f"pool_{i}": role.originator_importance for i, role in enumerate(roles)}

    # Main pool of each resource: the one where it executed most events (ties: the larger pool id).
    main_pool = {r: max(work, key=lambda p: (work[p].get(r, 0), p)) for r in log[RES].unique()}

    pools = []
    for i, role in enumerate(roles):
        pid = f"pool_{i}"
        pools.append({"id": pid, "activities": sorted(role.activities),
                      "members": sorted(r for r, p in main_pool.items() if p == pid),  # -> calendar
                      "all_resources": sorted(role.originator_importance)})            # -> who can execute
    return pools


# =====================================================================================================
# 2. Calendars
# =====================================================================================================
def discover_calendars(log, pools, coverage):
    """{pool id: Prosimos time periods}, from the completions of the pool's members (pools without
    members get no calendar: no resource uses it)."""
    # A calendar describes when the members work, so it uses all their events, of any activity.
    return {p["id"]: weekly_calendar(log.loc[log[RES].isin(p["members"]), END], coverage)
            for p in pools if p["members"]}


def weekly_calendar(timestamps, coverage):
    """Fracca et al.: "the day of the week and the hour of the day in which each role most frequently
    completed tasks", made explicit: the most frequent weekdays are kept until they cover `coverage` of the
    timestamps, then the same for the hours (on all days together); the calendar is kept weekdays x kept
    hours. Hours are those of the timestamps (UTC), as the simulation."""
    days = most_frequent_covering(timestamps.dt.weekday, coverage)  # e.g. [0, 1, 2, 3, 4] = Monday-Friday
    hours = most_frequent_covering(timestamps.dt.hour, coverage)    # e.g. [8, 9, 10, 14, 15]

    # One Prosimos time period per kept day and block of consecutive kept hours.
    periods = []
    for d in days:
        for begin, end in contiguous_blocks(hours):  # e.g. [(8, 11), (14, 16)]
            periods.append({"from": DAYS[d], "to": DAYS[d], "beginTime": f"{begin:02d}:00:00",
                            # 24:00:00 is not a valid time: a block ending at midnight ends at 23:59:59.999
                            "endTime": f"{end:02d}:00:00" if end < 24 else "23:59:59.999"})
    return periods


def most_frequent_covering(values, coverage):
    """The most frequent values, taken from the most frequent down, until they cover `coverage` of `values`.
    E.g. weekday counts Mon 30, Tue 25, Wed 25, Thu 10, Fri 8, Sat 2 -> cumulative shares 0.30, 0.55, 0.80,
    0.90, 0.98, 1.00 -> with coverage 0.95 Monday-Friday are kept (Friday is the first reaching 0.95)."""
    counts = values.value_counts().sort_values(ascending=False, kind="stable")  # most frequent first
    share = (counts.cumsum() / counts.sum()).values                             # cumulative share
    # first position whose cumulative share reaches `coverage` (-1e-12: a share equal to it up to rounding)
    last = int(np.searchsorted(share, coverage - 1e-12))
    return sorted(counts.index[: last + 1])


def contiguous_blocks(hours):
    """Sorted hours -> blocks [begin, end) of consecutive hours: [3, 4, 5, 9] -> [(3, 6), (9, 10)]"""
    blocks = []
    for h in hours:
        if blocks and blocks[-1][1] == h:        # h continues the last block: extend it by one hour
            blocks[-1] = (blocks[-1][0], h + 1)
        else:                                    # otherwise h opens a new block
            blocks.append((h, h + 1))
    return blocks


# =====================================================================================================
# 3. Inter-arrival time
# =====================================================================================================
def discover_inter_arrival_time(log, coverage, seed):
    """The arrival of a case is its first completion. Arrival calendar: weekly_calendar of the arrivals.
    Inter-arrival times: differences between consecutive arrivals minus the closed time of the arrival
    calendar between them (nights, weekends), since no case can arrive then. Approximation: Prosimos adds a
    sampled inter-arrival time in elapsed time and, if the result falls outside the calendar, moves it to the
    next opening, so near the closing time it can generate slightly shorter gaps than the real ones; without
    removing the closed time, the sample would contain the night and weekend gaps and be much further off.
    Returns {"calendar", "distribution": (name, [params])}."""
    arrivals = log.groupby(CASE)[END].min().sort_values()   # one arrival per case, in time order
    calendar = weekly_calendar(arrivals, coverage)           # when cases arrive
    open_slots = open_hours(calendar)                        # {(weekday, hour)} open in the calendar

    # inter-arrival time = time between two consecutive arrivals - closed time between them
    # (e.g. Mon-Fri 9-17: Friday 16:30 -> Monday 10:15 is 65.75 h - 64 h closed = 1.75 h)
    inter_arrival = [(b - a).total_seconds() - closed_seconds_between(a, b, open_slots)
                     for a, b in zip(arrivals.iloc[:-1], arrivals.iloc[1:])]  # n_cases - 1 values, in seconds

    distribution = find_best_fit_distribution(inter_arrival, np.random.default_rng(seed))
    return {"calendar": calendar, "distribution": distribution}


def open_hours(periods):
    """Prosimos time periods -> set of the open (weekday, hour) pairs, e.g. Monday 9:00-11:00 -> {(0, 9), (0, 10)}."""
    slots = set()
    for p in periods:
        end = 24 if p["endTime"].startswith("23:59") else int(p["endTime"][:2])
        for h in range(int(p["beginTime"][:2]), end):
            slots.add((DAYS.index(p["from"]), h))
    return slots


def closed_seconds_between(a, b, open_slots):
    """Seconds between the instants a and b that fall in closed hours of the calendar, counted one clock hour
    at a time (the first and last hours only for their part between a and b)."""
    closed, t = 0.0, a
    while t < b:
        step_end = min(t.floor("h") + pd.Timedelta(hours=1), b)  # end of the current hour, or b
        if (t.weekday(), t.hour) not in open_slots:
            closed += (step_end - t).total_seconds()
        t = step_end
    return closed


# =====================================================================================================
# 4. Branching probabilities
# =====================================================================================================
def discover_branching_probabilities(log, bpmn):
    """{gateway id: {flow id: probability}} for every XOR split. In the BPMN of the fitness-1 net
    ("activity defines place") a split follows the tasks of one activity X and each outgoing flow leads to a
    task of a next activity Y or to the end, so
        P(X -> Y) = n(X directly followed by Y) / n(X)
    (each trace has a single path in the net: counting equals replaying the log)."""
    # followed[(X, Y)] = how many times Y directly follows X in a case (Y = CASE_END after the last event,
    # X = CASE_START before the first one)
    next_activity = log.groupby(CASE)[ACT].shift(-1).fillna(CASE_END)
    followed = Counter(zip(log[ACT], next_activity))
    followed.update(Counter((CASE_START, a) for a in log.groupby(CASE)[ACT].first()))

    probabilities = {}
    for g in bpmn["splits"]:
        x = bpmn["split_after"][g]                                                 # activity before the split
        counts = {f: followed[(x, y)] for f, y in bpmn["split_flows"][g].items()}  # flow -> n(X -> Y)
        total = sum(counts.values())                                               # = n(X)
        # (total = 0 only if X never occurs, i.e. never in the simulation either: any split works)
        probabilities[g] = {f: (n / total if total else 1 / len(counts)) for f, n in counts.items()}
    return probabilities


def read_bpmn(bpmn_path):
    """What the JSON needs from the BPMN:
      "tasks":       {activity: [task ids]} (the fitness-1 net has one task per observed step X -> Y)
      "splits":      [ids of the XOR gateways with more than one outgoing flow]
      "split_after": {split id: the activity X whose tasks lead into it, or CASE_START}
      "split_flows": {split id: {outgoing flow id: next activity Y, or CASE_END if it leads to the end}}"""
    root = ET.parse(bpmn_path).getroot()

    # Nodes: id -> type (task, gateway, start/end event) and id -> name (the activity, for tasks)
    kind, name = {}, {}
    for e in root.iter():
        tag = e.tag.split("}")[-1]  # strip the XML namespace
        if tag in ("task", "exclusiveGateway", "parallelGateway", "inclusiveGateway", "startEvent", "endEvent"):
            kind[e.get("id")], name[e.get("id")] = tag, e.get("name") or ""
    if any(k in ("parallelGateway", "inclusiveGateway") for k in kind.values()):
        raise ValueError("only XOR gateways are supported (the fitness-1 net is a DFG)")

    # Arcs: node -> [(flow id, target node)] and node -> [source nodes]
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for e in root.iter():
        if e.tag.endswith("}sequenceFlow"):
            outgoing[e.get("sourceRef")].append((e.get("id"), e.get("targetRef")))
            incoming[e.get("targetRef")].append(e.get("sourceRef"))

    def leads_to_end(node):  # the end event, or gateways that only lead to it
        return kind[node] == "endEvent" or (kind[node] == "exclusiveGateway" and all(leads_to_end(t) for _, t in outgoing[node]))

    tasks = defaultdict(list)
    for n, k in kind.items():
        if k == "task":
            tasks[name[n]].append(n)

    splits = [n for n, k in kind.items() if k == "exclusiveGateway" and len(outgoing[n]) > 1]
    split_after, split_flows = {}, {}
    for g in splits:
        # all the nodes entering the split must be tasks of one activity (or the start event)
        before = {CASE_START if kind[s] == "startEvent" else name[s] for s in incoming[g]}
        if any(kind[s] not in ("task", "startEvent") for s in incoming[g]) or len(before) != 1:
            raise ValueError(f"gateway {g}: cannot tell which activity it follows")
        split_after[g] = before.pop()
        split_flows[g] = {}
        for f, t in outgoing[g]:
            if kind[t] == "task":
                split_flows[g][f] = name[t]
            elif leads_to_end(t):
                split_flows[g][f] = CASE_END
            else:
                raise ValueError(f"gateway {g}: flow {f} leads to an unsupported element")
    return {"tasks": dict(tasks), "splits": splits, "split_after": split_after, "split_flows": split_flows}


# =====================================================================================================
# The JSON
# =====================================================================================================
def write_simulation_parameters(json_path, bpmn, pools, calendars, arrivals, branching):
    """Prosimos format. Every resource of every role containing an activity can execute its tasks; the
    durations are placeholders ("fix" 0), overwritten by the estimator."""
    log_activities = {a for p in pools for a in p["activities"]}
    missing = log_activities - set(bpmn["tasks"])
    if missing:
        raise ValueError(f"activities of the log not in the BPMN: {sorted(missing)}")

    # can_execute[r] = activities of all the roles in which r appears (not only of its main pool)
    can_execute = defaultdict(set)
    for p in pools:
        for r in p["all_resources"]:
            can_execute[r] |= set(p["activities"])

    def distribution(name, params):
        return {"distribution_name": name, "distribution_params": [{"value": v} for v in params]}

    parameters = {
        "model_type": "CRISP",
        # 1. pools: each resource listed once, in its main pool, with its calendar and the tasks it can execute
        "resource_profiles": [
            {"id": p["id"], "name": ", ".join(p["activities"]),
             "resource_list": [{"id": r, "name": r, "cost_per_hour": 1, "amount": 1, "calendar": f"{p['id']}_calendar",
                                "assigned_tasks": sorted(t for a in can_execute[r] for t in bpmn["tasks"][a])}
                               for r in p["members"]]}
            for p in pools if p["members"]],
        # 2. calendars, one per pool with members
        "resource_calendars": [{"id": f"{pid}_calendar", "name": f"{pid}_calendar", "time_periods": periods}
                               for pid, periods in calendars.items()],
        # who can execute each task (Prosimos picks the one that becomes free first) and how long it takes
        "task_resource_distribution": [
            {"task_id": t, "resources": [{"resource_id": r, **distribution("fix", [0])}
                                         for r in sorted(can_execute) if a in can_execute[r]]}
            for a, ts in bpmn["tasks"].items() for t in ts],
        # 3. arrivals
        "arrival_time_distribution": distribution(*arrivals["distribution"]),
        "arrival_time_calendar": arrivals["calendar"],
        # 4. branching probabilities
        "gateway_branching_probabilities": [
            {"gateway_id": g, "probabilities": [{"path_id": f, "value": p} for f, p in flows.items()]}
            for g, flows in branching.items()],
        "event_distribution": [],
        "batch_processing": [],
    }
    Path(json_path).write_text(json.dumps(parameters, indent=2))
    return json_path
