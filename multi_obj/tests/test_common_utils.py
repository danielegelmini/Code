import unittest
from datetime import datetime

import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking

from prosit.simulator import SimulatorEngine, SimulatorParameters
from prosit.utils.common_utils import return_enabled_transitions
from utils.simulation_functions import build_recommender_df


class ReturnEnabledTransitionsTests(unittest.TestCase):
    def test_respects_arc_weights(self):
        net = PetriNet("test")
        p1 = PetriNet.Place("p1")
        p2 = PetriNet.Place("p2")
        t = PetriNet.Transition("t")

        net.places.add(p1)
        net.places.add(p2)
        net.transitions.add(t)

        arc = PetriNet.Arc(source=p1, target=t)
        arc.weight = 2
        p1._Place__out_arcs.add(arc)
        t._Transition__in_arcs.add(arc)
        net._PetriNet__arcs.add(arc)

        marking = Marking()
        marking[p1] = 1
        self.assertNotIn(t, return_enabled_transitions(net, marking))

        marking[p1] = 2
        self.assertIn(t, return_enabled_transitions(net, marking))


class SimulatorEngineRecommendationTests(unittest.TestCase):
    def test_emits_recommendation_even_when_only_invisible_path_exists(self):
        net = PetriNet("test")
        p0 = PetriNet.Place("p0")
        p1 = PetriNet.Place("p1")
        p2 = PetriNet.Place("p2")
        t_inv = PetriNet.Transition("t_inv")
        t_rec = PetriNet.Transition("t_rec")
        t_rec.label = "Recommend"

        net.places.add(p0)
        net.places.add(p1)
        net.places.add(p2)
        net.transitions.add(t_inv)
        net.transitions.add(t_rec)

        arc_inv_in = PetriNet.Arc(source=p0, target=t_inv)
        arc_inv_out = PetriNet.Arc(source=t_inv, target=p1)
        arc_rec_in = PetriNet.Arc(source=p1, target=t_rec)
        arc_rec_out = PetriNet.Arc(source=t_rec, target=p2)

        for arc in (arc_inv_in, arc_inv_out, arc_rec_in, arc_rec_out):
            net._PetriNet__arcs.add(arc)

        p0._Place__out_arcs.add(arc_inv_in)
        t_inv._Transition__in_arcs.add(arc_inv_in)
        t_inv._Transition__out_arcs.add(arc_inv_out)
        p1._Place__in_arcs.add(arc_inv_out)
        p1._Place__out_arcs.add(arc_rec_in)
        t_rec._Transition__in_arcs.add(arc_rec_in)
        t_rec._Transition__out_arcs.add(arc_rec_out)
        p2._Place__in_arcs.add(arc_rec_out)

        initial_marking = Marking()
        initial_marking[p0] = 1
        final_marking = Marking()
        final_marking[p2] = 1

        params = SimulatorParameters(net, initial_marking, final_marking)
        engine = SimulatorEngine(params)
        case = {
            "rec_act": "Recommend",
            "marking": initial_marking,
            "enabled": {t_inv: datetime.now()},
            "arrival_time": datetime.now(),
            "place_token_time": {p0: datetime.now(), p1: None, p2: None},
            "pending_invisible_path": [],
        }

        chosen_transition, activity, _, flag_rec = engine._resolve_recommended_transition(case, [t_inv])

        self.assertTrue(flag_rec)
        self.assertIsNone(chosen_transition)
        self.assertEqual(activity, "Recommend")

    def test_build_recommender_df_attaches_recommendation_to_last_chronological_event(self):
        prev_log = pd.DataFrame(
            [
                {
                    "case:concept:name": "case_1",
                    "concept:name": "A",
                    "org:resource": "R1",
                    "start:timestamp": "2024-01-01 00:00:00",
                    "time:timestamp": "2024-01-01 00:01:00",
                },
                {
                    "case:concept:name": "case_1",
                    "concept:name": "B",
                    "org:resource": "R2",
                    "start:timestamp": "2024-01-01 00:01:00",
                    "time:timestamp": "2024-01-01 00:02:00",
                },
            ]
        )
        prev_log["time:timestamp"] = pd.to_datetime(prev_log["time:timestamp"])
        prev_log["start:timestamp"] = pd.to_datetime(prev_log["start:timestamp"])

        prev_log = prev_log.iloc[[1, 0]].copy().reset_index(drop=True)

        recommendations = {"case_1": {"act": "Recommend", "res": "R3"}}
        updated_log = build_recommender_df(prev_log, recommendations)

        latest_row = updated_log.loc[updated_log["time:timestamp"].idxmax()]
        self.assertEqual(latest_row["concept:name"], "B")
        self.assertEqual(latest_row["recommendation:act"], "Recommend")
        self.assertEqual(latest_row["recommendation:res"], "R3")


if __name__ == "__main__":
    unittest.main()
