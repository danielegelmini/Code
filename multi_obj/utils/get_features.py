from typing import Tuple, List
import pandas as pd
import joblib

def load_case_study(case_study):
    data_path = f"./case_studies/{case_study}/"
    train_data = pd.read_csv(data_path + "train_data.csv") # Training data
    test_data = pd.read_csv(data_path + "test_log_with_last_act.csv") # Only contains query instances (last event) at split time
    test_log = pd.read_csv(data_path + "test_log.csv") # All prefixes in test data
    return train_data, test_data, test_log

def get_case_study_features(case_study):
    # Loading predictive model
    predictive_outcome_model = joblib.load(f'./case_studies/{case_study}/model/catboost_model_label.joblib')
    predictive_time_model = joblib.load(f'./case_studies/{case_study}/model/catboost_model_sigmoid_mm.joblib')
    # Loading features
    case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove = get_features(case_study)
    return predictive_outcome_model, predictive_time_model, case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove

def get_features(case_study: str) -> Tuple[str, str, str, List[str], List[str], List[str]]:
    """
        Return the feature configuration for a given case study.
    """
    # Column name constants
    case_id_name = "case:concept:name"
    activity_column_name = "concept:name"
    end_date_name = "time:timestamp"
    start_date_name = "start:timestamp"
    resource_column_name = "org:resource"

    # Same for all
    columns_to_remove = [
        case_id_name, start_date_name, end_date_name, "total_time",  "label", "sigmoid_mm", "remaining_time"
    ]
    # Use the case_study directly so before/after can have distinct configs
    key = case_study


    CONFIG = {
        # --- BPI 2017 (before variant: unprefixed columns) ---
        "bpi17_before": {
            "continuous": [
                "RequestedAmount",
                "# ACTIVITY=A_Cancelled", "# ACTIVITY=O_Returned",
                "# ACTIVITY=A_Denied", "# ACTIVITY=A_Submitted",
                "# ACTIVITY=O_Cancelled", "# ACTIVITY=O_Refused",
                "# ACTIVITY=W_Validate application",
                "# ACTIVITY=W_Assess potential fraud",
                "# ACTIVITY=W_Complete application", "# ACTIVITY=A_Complete",
                "# ACTIVITY=W_Call after offers", "# ACTIVITY=O_Sent (online only)",
                "# ACTIVITY=O_Created", "# ACTIVITY=O_Sent (mail and online)",
                "# ACTIVITY=A_Validating", "# ACTIVITY=O_Accepted",
                "# ACTIVITY=W_Call incomplete files", "# ACTIVITY=A_Accepted",
                "# ACTIVITY=A_Create Application", "# ACTIVITY=A_Concept",
                "# ACTIVITY=W_Handle leads", "# ACTIVITY=A_Pending",
                "# ACTIVITY=A_Incomplete", "# ACTIVITY=O_Create Offer",
                "time_from_start", "time_from_previous_event(start)", "event_duration",
            ],
            "categorical": [
                activity_column_name, resource_column_name, "NEXT_ACTIVITY",
                "NEXT_RESOURCE", "weekday", "Action", "EventOrigin",
                "LoanGoal", "ApplicationType",
            ],
        },

        # --- BPI 2017 (after variant: prefixed columns) ---
        "bpi17_after": {
            "continuous": [
                "case:RequestedAmount",
                "# ACTIVITY=A_Cancelled", "# ACTIVITY=O_Returned",
                "# ACTIVITY=A_Denied", "# ACTIVITY=A_Submitted",
                "# ACTIVITY=O_Cancelled", "# ACTIVITY=O_Refused",
                "# ACTIVITY=W_Validate application",
                "# ACTIVITY=W_Assess potential fraud",
                "# ACTIVITY=W_Complete application", "# ACTIVITY=A_Complete",
                "# ACTIVITY=W_Call after offers", "# ACTIVITY=O_Sent (online only)",
                "# ACTIVITY=O_Created", "# ACTIVITY=O_Sent (mail and online)",
                "# ACTIVITY=A_Validating", "# ACTIVITY=O_Accepted",
                "# ACTIVITY=W_Call incomplete files", "# ACTIVITY=A_Accepted",
                "# ACTIVITY=A_Create Application", "# ACTIVITY=A_Concept",
                "# ACTIVITY=W_Handle leads", "# ACTIVITY=A_Pending",
                "# ACTIVITY=A_Incomplete", "# ACTIVITY=O_Create Offer",
                "time_from_start", "time_from_previous_event(start)", "event_duration",
            ],
            "categorical": [
                activity_column_name, resource_column_name, "NEXT_ACTIVITY",
                "NEXT_RESOURCE", "weekday", "Action", "EventOrigin",
                "case:LoanGoal", "case:ApplicationType",
            ],
        },
        
        # --- BPI 2012 ---
        "BPI12": {
            "continuous": [
                "AMOUNT_REQ",
                "# ACTIVITY=A_REGISTERED", "# ACTIVITY=O_CREATED",
                "# ACTIVITY=A_ACTIVATED", "# ACTIVITY=A_PREACCEPTED",
                "# ACTIVITY=O_ACCEPTED", "# ACTIVITY=W_Completeren aanvraag",
                "# ACTIVITY=W_Nabellen incomplete dossiers", "# ACTIVITY=O_CANCELLED",
                "# ACTIVITY=O_DECLINED", "# ACTIVITY=A_FINALIZED",
                "# ACTIVITY=A_APPROVED", "# ACTIVITY=A_SUBMITTED", "# ACTIVITY=O_SENT",
                "# ACTIVITY=W_Valideren aanvraag", "# ACTIVITY=W_Afhandelen leads",
                "# ACTIVITY=A_DECLINED", "# ACTIVITY=A_PARTLYSUBMITTED",
                "# ACTIVITY=A_ACCEPTED", "# ACTIVITY=O_SENT_BACK",
                "# ACTIVITY=A_CANCELLED", "# ACTIVITY=O_SELECTED",
                "# ACTIVITY=W_Beoordelen fraude", "# ACTIVITY=W_Nabellen offertes",
                "time_from_start", "time_from_previous_event(start)", "event_duration",
            ],
            "categorical": [
                activity_column_name, resource_column_name, "NEXT_ACTIVITY", "NEXT_RESOURCE", "weekday",
            ],
        },
        
        # --- BAC ---
        "BAC": {
            "continuous": [
                "# ACTIVITY=Pending Request for Reservation Closure",
                "# ACTIVITY=Pending Request for Network Information",
                "# ACTIVITY=Evaluating Request (NO registered letter)",
                "# ACTIVITY=Pending Request for acquittance of heirs",
                "# ACTIVITY=Service closure Request with BO responsibility",
                "# ACTIVITY=Authorization Requested",
                "# ACTIVITY=Evaluating Request (WITH registered letter)",
                "# ACTIVITY=Back-Office Adjustment Requested",
                "# ACTIVITY=Service closure Request with network responsibility",
                "# ACTIVITY=Request completed with customer recovery",
                "# ACTIVITY=Request deleted", "# ACTIVITY=Request created",
                "# ACTIVITY=Pending Liquidation Request",
                "# ACTIVITY=Request completed with account closure",
                "# ACTIVITY=Network Adjustment Requested",
                "time_from_start", "time_from_previous_event(start)", "event_duration",
            ],
            "categorical": [
                activity_column_name, resource_column_name, "NEXT_ACTIVITY",
                "NEXT_RESOURCE", "weekday", "CLOSURE_TYPE", "CLOSURE_REASON", "org:role",
            ],
        },
    }

    # BPI12_sim is the fully-simulated stand-in for BPI12 (see
    # 9_generate_simulated_training_set.py); it shares BPI12's activity/resource
    # schema, so it reuses BPI12's feature configuration verbatim.
    CONFIG["BPI12_sim"] = CONFIG["BPI12"]

    # BPI12_reordered/BPI12_reordered_sim are rebuilt from the "clean" BPI12 raw log
    # (see 0_prepare_bpi12_clean_log.py), which uses different, more granular English
    # names for the 6 W_ activities instead of BPI12's original Dutch ones -- same
    # underlying process step, different label. Everything else (A_/O_ activities,
    # AMOUNT_REQ, resource/time features) is identical to BPI12, so this copies
    # BPI12's continuous-feature list and only swaps those 6 names (mapping verified
    # during the reordering investigation: same activity, same position in the net,
    # in most cases confirmed by matching real COMPLETE timestamps 1:1 between the
    # two logs).
    _BPI12_TO_REORDERED_ACTIVITY = {
        "# ACTIVITY=W_Completeren aanvraag": "# ACTIVITY=W_Complete_preaccepted_appl",
        "# ACTIVITY=W_Nabellen offertes": "# ACTIVITY=W_Call_after_offer",
        "# ACTIVITY=W_Valideren aanvraag": "# ACTIVITY=W_Assess_application",
        "# ACTIVITY=W_Afhandelen leads": "# ACTIVITY=W_Fix_incoplete_submission",
        "# ACTIVITY=W_Beoordelen fraude": "# ACTIVITY=W_Assess_fraud",
        "# ACTIVITY=W_Nabellen incomplete dossiers": "# ACTIVITY=W_Call_missing_information",
    }
    _reordered_config = {
        "continuous": [
            _BPI12_TO_REORDERED_ACTIVITY.get(f, f) for f in CONFIG["BPI12"]["continuous"]
        ],
        "categorical": list(CONFIG["BPI12"]["categorical"]),
    }
    CONFIG["BPI12_reordered"] = _reordered_config
    CONFIG["BPI12_reordered_sim"] = _reordered_config

    if key not in CONFIG:
        raise ValueError(f"Unknown case_study: {case_study!r}")

    cfg = CONFIG[key]
    return (
        case_id_name,
        activity_column_name,
        resource_column_name,
        cfg["continuous"],
        cfg["categorical"],
        columns_to_remove,
    )
