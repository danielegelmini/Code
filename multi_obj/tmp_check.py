import pickle
from pathlib import Path
import pandas as pd
import pm4py
from utils.simulation_functions import build_recommender_df
from prosit.simulator import SimulatorParameters, SimulatorEngine
from prosit.utils.common_utils import return_enabled_transitions

base = Path('.')
case_study='BAC'
case_dir=base/'case_studies'/case_study
prev_log = pd.read_csv(case_dir/'test_log.csv', parse_dates=['time:timestamp','start:timestamp'])
pkl_path = case_dir/'recommendations'/f'recommendations_{case_study}_nsga2.pkl'
rec_dict = pickle.load(open(pkl_path,'rb'))
recommendations = {case_id:{'act':rec[0],'res':rec[1]} for case_id,rec in rec_dict.items()}
log_rec = build_recommender_df(prev_log.copy(), recommendations)
case_id = 201812007788
prefix_log_c = log_rec[log_rec['case:concept:name']==case_id]
print('recommendation tail')
print(prefix_log_c[['concept:name','recommendation:act','recommendation:res']].tail(3).to_string(index=False))
print('last rec act', prefix_log_c['recommendation:act'].iloc[-1])
print('last rec res', prefix_log_c['recommendation:res'].iloc[-1])
print('rows', len(prefix_log_c))

net, im, fm = pm4py.read_pnml(str(case_dir/'discovery_output'/f'model_{case_study}.pnml'))
params = SimulatorParameters(net, im, fm)
params.from_json(str(case_dir/'discovery_output'/f'simulator_params_{case_study}.json'))
engine = SimulatorEngine(params)

replayed = pm4py.algorithms.conformance.token_replay.apply(prefix_log_c, params.net, params.initial_marking, params.final_marking)
current_marking_c = replayed[0]['reached_marking']
print('reached marking', current_marking_c)
enabled = return_enabled_transitions(net, current_marking_c)
print('enabled labels', [t.label for t in enabled])

case = {
    'arrival_time': prefix_log_c['time:timestamp'].iloc[-1],
    'case_id': 0,
    'marking': current_marking_c,
    'place_token_time': {},
    'enabled': {},
    'history': {t: 0 for t in params.net_transition_labels},
    'attributes': {},
    'rec_act': prefix_log_c['recommendation:act'].iloc[-1],
    'rec_res': prefix_log_c['recommendation:res'].iloc[-1],
    'pending_invisible_path': []
}
for place in net.places:
    case['place_token_time'][place] = None
for place in current_marking_c.keys():
    case['place_token_time'][place] = case['arrival_time']
for t in enabled:
    input_places = [arc.source for arc in net.arcs if arc.target == t]
    enabled_time = max(case['place_token_time'][p] for p in input_places)
    case['enabled'][t] = prefix_log_c['time:timestamp'].iloc[-1]
print('resolve', engine._resolve_recommended_transition(case, list(enabled)))
