import pandas as pd
import pm4py
import pickle
from pathlib import Path
from utils.simulation_functions import build_recommender_df
from prosit.simulator import SimulatorParameters, SimulatorEngine
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

base = Path('c:/Users/Utente/Desktop/tesi magistrale/Code/multi_obj')
case_study = 'BAC'
case_dir = base / 'case_studies' / case_study
prev_log = pd.read_csv(case_dir / 'test_log.csv', parse_dates=['time:timestamp', 'start:timestamp'])
net, im, fm = pm4py.read_pnml(str(case_dir / 'discovery_output' / f'model_{case_study}.pnml'))
params = SimulatorParameters(net, im, fm)
params.from_json(str(case_dir / 'discovery_output' / f'simulator_params_{case_study}.json'))

pkl_path = case_dir / 'recommendations' / f'recommendations_{case_study}_nsga2.pkl'
with open(pkl_path, 'rb') as fh:
    rec_dict = pickle.load(fh)
recs = {case_id: {'act': rec[0], 'res': rec[1]} for case_id, rec in rec_dict.items()}
log_rec = build_recommender_df(prev_log.copy(), recs)
case_id = 201812007788
prefix_log_c = log_rec[log_rec['case:concept:name'] == case_id]
print(prefix_log_c[['concept:name','recommendation:act','recommendation:res']].tail(10).to_string(index=False))
print('rec last', prefix_log_c['recommendation:act'].iloc[-1])
replayed = token_replay.apply(prefix_log_c, net, im, fm)
print('replayed len', len(replayed))
if replayed:
    print('reached_marking', replayed[0]['reached_marking'])
    print('activated_transitions', [t.label for t in replayed[0]['activated_transitions'] if t.label])

# create engine and inspect initial case behavior
engine = SimulatorEngine(params)
case = {
    'arrival_time': prefix_log_c['time:timestamp'].iloc[-1],
    'case_id': 0,
    'marking': replayed[0]['reached_marking'],
    'place_token_time': {},
    'enabled': {},
    'history': {t:0 for t in params.net_transition_labels},
    'attributes': {},
    'rec_act': prefix_log_c['recommendation:act'].iloc[-1],
    'rec_res': prefix_log_c['recommendation:res'].iloc[-1],
    'pending_invisible_path': []
}
for place in net.places:
    case['place_token_time'][place] = None
for place in replayed[0]['reached_marking'].keys():
    case['place_token_time'][place] = case['arrival_time']
from prosit.utils.common_utils import return_enabled_transitions
enabled = return_enabled_transitions(net, case['marking'])
for t in enabled:
    input_places = [arc.source for arc in net.arcs if arc.target == t]
    enabled_time = max(case['place_token_time'][p] for p in input_places)
    case['enabled'][t] = prefix_log_c['time:timestamp'].iloc[-1]
print('enabled labels', [t.label for t in enabled])
print('resolved', engine._resolve_recommended_transition(case, enabled))
