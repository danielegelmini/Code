from prosit.utils.rule_utils import DecisionRules
from pm4py.objects.petri_net.obj import PetriNet
import copy
from prosit.utils.distribution_utils import sampling_from_dist
import scipy.stats as stats


def _make_json_safe(obj):
    if isinstance(obj, dict):
        safe_obj = {}
        for k, v in obj.items():
            if k == "sampled":
                continue
            safe_key = k
            if isinstance(k, tuple):
                safe_key = str(k)
            elif isinstance(k, (list, set)):
                safe_key = str(k)
            safe_obj[safe_key] = _make_json_safe(v)
        return safe_obj

    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, set):
        return [_make_json_safe(v) for v in sorted(obj, key=lambda x: str(x))]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "__module__") and obj.__module__.startswith("scipy.stats"):
        return obj.name

    return str(obj)


def transition_to_name(t: PetriNet.Transition) -> str:

    if t.label is not None:
        name = t.label
    else:
        name = t.name

    return name


def name_to_transition(s: str, net: PetriNet) -> PetriNet.Transition:

    for t in net.transitions:
        if s == t.name:
            return t
        if s == t.label:
            return t


def decision_rules_to_dict(d: DecisionRules) -> dict:

    def convert(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "sampled":
                    continue  # Remove the 'sampled' key
                safe_key = k
                if isinstance(k, tuple):
                    safe_key = str(k)
                elif isinstance(k, (list, set)):
                    safe_key = str(k)
                new_obj[safe_key] = convert(v)
            return new_obj
        elif isinstance(obj, tuple):
            return {"dist_name": convert(obj[0]), "params": _make_json_safe(obj[1]), "min_value": obj[2], "max_value": obj[3]}

        elif hasattr(obj, '__module__') and obj.__module__.startswith("scipy.stats"):
            return obj.name  # Convert scipy distribution to string
        else:
            return _make_json_safe(obj)

    if isinstance(d, tuple):
        return {"dist_name": convert(d[0]), "params": _make_json_safe(d[1]), "min_value": d[2], "max_value": d[3], "mean_value": d[4]}

    if not isinstance(d, DecisionRules):
        if d is None:
            d = 1
        return _make_json_safe(d)

    return convert(copy.deepcopy(d.rules))


def convert_calendar_names(calendar: dict, to_number = False):

    if to_number:
        weekday_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
    else:
        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }

    new_calendar = {}
    for weekday, hours in calendar.items():
        new_calendar[weekday_map[weekday]] = {int(k): v for k, v in hours.items()}

    return new_calendar


def dict_to_decrules(d: dict) -> DecisionRules:

    def convert(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "true":
                    new_k = True
                elif k == "false":
                    new_k = False
                else:
                    new_k = k
                    try:
                        new_k = int(k)
                    except:
                        pass
                if k == "dist":
                    new_obj[k] = fromstr_to_scipy(obj[k]["dist_name"]), tuple(obj[k]["params"]), obj[k]["min_value"], obj[k]["max_value"]
                    new_obj["sampled"] = sampling_from_dist(new_obj[k][0], new_obj[k][1], new_obj[k][2], new_obj[k][3], new_obj["value"])
                else:
                    new_obj[new_k] = convert(v)
            return new_obj
        else:
            return obj

    dr = DecisionRules()
    dr.rules = convert(d)
    return dr
    

def fromstr_to_scipy(s):
    if s == "fixed":
        return s
    else:
        return getattr(stats, s)
