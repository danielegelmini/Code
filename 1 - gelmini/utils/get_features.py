from typing import Tuple, List
import pandas as pd
import joblib

def load_case_study(case_study):
    data_path = f"./case_studies/{case_study}/"
    train_data = pd.read_csv(data_path + "train_data.csv") # Training data
    test_data = pd.read_csv(data_path + "test_log_with_last_act_most.csv") # Only contains query instances (last event) at split time
    test_log = pd.read_csv(data_path + "test_log.csv") # All prefixes in test data
    return train_data, test_data, test_log

def get_case_study_features(case_study):
    # Loading predictive model
    predictive_model = joblib.load(f'./case_studies/{case_study}/model/catboost_model.joblib') 
    # Loading features
    case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove = get_features(case_study)
    return predictive_model, case_id_name, activity_column_name, resource_column_name, continuous_features, categorical_features, columns_to_remove

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
        case_id_name, start_date_name, end_date_name,
        "total_time", "remaining_time_normalized", "status", "sigmoid", "outcome",  "outcome_sigmoid", "sigmoid_mm", "remaining_time"
    ]
    # Some case labels are aliases of the same schema
    aliases = {
        "bpi17_before": "bpi17",
        "bpi17_after": "bpi17",
    }
    key = aliases.get(case_study, case_study)


    CONFIG = {
        # --- BPI 2017 (before/after share the same features) ---
        "bpi17": {
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
        
        # --- BPI 2012 ---
        "bpi12": {
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
