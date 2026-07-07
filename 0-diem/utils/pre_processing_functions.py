from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statistics import median
import os, json, random
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle
from datetime import datetime
import pm4py


def linear_combination(df, lambda_weight):
    # Compute linear combination for normalized remaining time and case outcome
    df['outcome'] = lambda_weight * (1 - df['label']) + (1 - lambda_weight) * df['sigmoid_mm']
    return df

def data_pre_processing(case_study, case_id_position, start_date_position, date_format, end_date_position, case_id_name, 
                        start_date_name, end_date_name, activity_column_name, resource_column_name):
    print("Loading data...")
    log = pm4py.read_xes(f'case_studies/log_{case_study}.xes')
    data = pm4py.convert_to_dataframe(log)

    ordered_cols = ["case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"]
    remaining_cols = [col for col in data.columns if col not in ordered_cols]
    data = data[ordered_cols + remaining_cols]

    # Function "format_dataframe" creates another column for start timestamp, case index, and event index
    df = pm4py.utils.format_dataframe(data, case_id=case_id_name, activity_key=activity_column_name, timestamp_key=end_date_name, start_timestamp_key=start_date_name) 
    df["time_timestamp"] = df[end_date_name] 
    df = df.drop(columns=["@@index", "@@case_index"], axis=1) # Dropping columns that repeat information

    print("...adding total time...")
    df = getting_total_time(df, case_id_name, start_date_name, end_date_name)

    print("...adding activity frequency...")
    df = preprocessing_activity_frequency(df, activity_column_name, case_id_name, start_date_name)
    
    print("...adding next activity and next resource...")
    df = add_next_act_res(df, activity_column_name, resource_column_name, case_id_name)
   
    print("...adding features...")
    # Calculate positions dynamic to the current data frame shape
    case_id_position = df.columns.get_loc("case:concept:name")
    start_date_position = df.columns.get_loc("start:timestamp")
    end_date_position = df.columns.get_loc("time:timestamp")
    
    df = prepare_data_and_add_features(df, case_id_position, start_date_position, 
                                       date_format, end_date_position)
    
    print("...adding outcome label...")
    df = data_labelling(df, case_study)

    # Remove cases with less than 3 events (not meaningful for recommendation purpose)
    df = df[df.groupby(case_id_name)[case_id_name].transform('count') > 3].reset_index(drop=True)

    df = df.rename(columns={"time_timestamp": "time:timestamp", "start_timestamp": "start:timestamp", "leadtime":"total_time"})
    del df['time_from_midnight']
    output_dir = Path(f"./case_studies/{case_study}")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "preprocessed_data.csv", index=False)    
    
    return df

def convert_dtypes_bpi12(df, mode):
    if mode == 'experiment':
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        df['NEXT_RESOURCE'] = df['NEXT_RESOURCE'].astype(str)
        df['org:resource'] = df['org:resource'].astype(str)
    elif mode == 'simulation_prep':
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        df['Next_resource'] = df['Next_resource'].astype(str)
    elif mode == 'simulation':
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        df['org:resource'] = df['org:resource'].astype(str)
    elif mode == 'simulation_':
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        df['res_1'] = df['res_1'].astype(str)
    return df

def convert_time(time_to_convert):
    dt = datetime.strptime(time_to_convert.split('+')[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
    return dt

def getting_remaining_time(dataframe, case_id_name, start_date_name, end_date_name):
    dataframe["remaining_time"] = ""
    dataframe['time:timestamp'] = dataframe['time:timestamp'].apply(convert_time)
    dataframe['start:timestamp'] = dataframe['start:timestamp'].apply(convert_time)

    list_trace_id = set(dataframe[case_id_name].unique()) # Getting list of trace ID
    for trace_id in list_trace_id:
        trace_df = dataframe[dataframe[case_id_name] == trace_id] # Getting sub df for each trace
        
        sub_trace_df_sorted = trace_df.sort_values(by=[end_date_name]) # Sorting by time
            
        indices = sub_trace_df_sorted.index.values.tolist()
        last_event_idx = indices[-1]

        for idx in indices:
            dataframe['remaining_time'][idx] = (sub_trace_df_sorted[end_date_name][last_event_idx] - sub_trace_df_sorted[end_date_name][idx]).total_seconds()

    dataframe = dataframe.sort_values(by=[end_date_name]).reset_index(drop=True)  
    return dataframe


def getting_total_time(dataframe, case_id_name, start_date_name, end_date_name): 
    """
    Calculate the total execution time for each case in the dataframe.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing event log data.
        case_id_name (str): The name of the column representing the case ID.
        start_date_name (str): The name of the column representing the start date of events.
        end_date_name (str): The name of the column representing the end date of events.
    
    Returns:
        pd.DataFrame: The dataframe with an additional column 'total_time'
                      representing the total execution time for each case in seconds.
    """
    # Ensure datetime
    dataframe[start_date_name] = pd.to_datetime(dataframe[start_date_name])
    dataframe[end_date_name] = pd.to_datetime(dataframe[end_date_name])

    # Compute per case: last end - first start
    total_times = (
        dataframe.groupby(case_id_name)
        .agg(first_start=(start_date_name, "min"), last_end=(end_date_name, "max"))
    )
    total_times["total_time"] = (total_times["last_end"] - total_times["first_start"]).dt.total_seconds().round()

    # Map back to original dataframe
    dataframe["total_time"] = dataframe[case_id_name].map(total_times["total_time"])

    return dataframe


def add_next_act(df, activity_column_name, case_id_name): 
    """
    Adds columns for the next activity to the dataframe for each case.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe containing the event log.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.
        case_id_name (str): The name of the column containing case IDs.
    
    Returns:
        pandas.DataFrame: The dataframe with two new column 'NEXT_ACTIVITY' indicating 
        the next activity and resource for each event in the case.
    """

    list_unique_id = df[case_id_name].unique() # Extracting list of cases
    idx = 0
    df['NEXT_ACTIVITY'] = ""
    for case_id in list_unique_id:
        num_activities = np.sum(df[case_id_name]==case_id) # Counting number of activities
        sub_df = df.loc[df[case_id_name] == case_id].reset_index(drop=True) # Creating a dataframe with all activities refered to the same case_id
        for i in range(num_activities):
            if i == num_activities - 1: # Indicating last activity
                df['NEXT_ACTIVITY'][idx] = 'end'
            else:
                df['NEXT_ACTIVITY'][idx] = sub_df[activity_column_name].loc[i+1] # Assigning next activity
            idx = idx +1
    return df

def add_next_act_res(df, activity_column_name, resource_column_name, case_id_name): 
    """
    Adds columns for the next activity and next resource to the dataframe for each case.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe containing the event log.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.
        case_id_name (str): The name of the column containing case IDs.
    
    Returns:
        pandas.DataFrame: The dataframe with two new columns 'NEXT_ACTIVITY' and 'NEXT_RESOURCE' 
                        indicating the next activity and resource for each event in the case.
    """

    list_unique_id = df[case_id_name].unique() # Extracting list of cases
    idx = 0
    df['NEXT_ACTIVITY'] = ""
    df['NEXT_RESOURCE'] = ""
    for case_id in list_unique_id:
        num_activities = np.sum(df[case_id_name]==case_id) # Counting number of activities
        sub_df = df.loc[df[case_id_name] == case_id].reset_index(drop=True) # Creating a dataframe with all activities refered to the same case_id
        for i in range(num_activities):
            if i == num_activities - 1: # Indicating last activity
                df['NEXT_ACTIVITY'][idx] = 'end'
                df['NEXT_RESOURCE'][idx] = 'end'
            else:
                df['NEXT_ACTIVITY'][idx] = sub_df[activity_column_name].loc[i+1] # Assigning next activity
                df['NEXT_RESOURCE'][idx] = sub_df[resource_column_name].loc[i+1] # Assigning next resource
            idx = idx +1
    return df

def preprocessing_activity_frequency(dataframe, activity_column_name, case_id_name, start_date_name):  
    """
    Preprocesses the given dataframe to calculate the frequency of each activity for each case.
    This function adds new columns to the dataframe, where each column represents the frequency of a specific activity 
    up to the current event in the trace. The columns are named in the format '# ACTIVITY=<activity_name>'.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing event log data.
        activity_column_name (str): The name of the column containing activity names.
        case_id_name (str): The name of the column containing case IDs.
        start_date_name (str): The name of the column containing the start date of the events.
    
    Returns:
        pd.DataFrame: The modified dataframe with additional columns for activity frequencies.
    """

    list_activities = set(dataframe[activity_column_name].unique()) # Getting list of activities
    list_trace_id = set(dataframe[case_id_name].unique()) # Getting list of trace ID

    for activity in list_activities:
        column_name = '# ' + 'ACTIVITY' + '=' + activity
        dataframe[column_name] = 0 # Creating empty columns for activity frequency

    for trace_id in list_trace_id:
        sub_trace_df = dataframe[dataframe[case_id_name] == trace_id] # Getting sub df for each trace
        sub_trace_df1 = sub_trace_df.copy()
        sub_trace_df_sorted = sub_trace_df1.sort_values(by=[start_date_name]) # Sorting by time

        indexes = sub_trace_df_sorted.index.values.tolist()
        start_event_idx = indexes[0]
        last_event_idx = indexes[-1]

        history = [] # List of previous activities
        for idx in indexes:
            if idx != start_event_idx: # Exclude start event
                previous_activity = dataframe[activity_column_name][idx-1]
                history.append(previous_activity)
                for activity in history:
                    if activity == previous_activity:
                        target_column = '# ' + 'ACTIVITY' + '=' + activity
                        dataframe[target_column][idx] = dataframe[target_column][idx-1] + 1
                    else:
                        target_column = '# ' + 'ACTIVITY' + '=' + activity
                        dataframe[target_column][idx] = dataframe[target_column][idx-1]
    return dataframe

def data_labelling(df, case_study):
    case_study = case_study.lower()

    if case_study == "bpi12":
        df["label"] = (
            df.groupby("case:concept:name")["concept:name"]
            .transform(lambda x: x.eq("O_ACCEPTED").any())
            .astype(int)
        )

    elif case_study == "bpi17_before" or case_study == "bpi17_after":
        df["label"] = (
            df.groupby("case:concept:name")["concept:name"]
            .transform(lambda x: x.eq("O_Accepted").any())
            .astype(int)
        )

    elif case_study == "bac":
        df["label"] = (
            df.groupby("case:concept:name")["concept:name"]
            .transform(lambda x: ~x.isin(["Back-Office Adjustment Requested", "Network Adjustment Requested"]).any())
            .astype(int)
        )
    return df

def calculateTimeFromMidnight(actual_datetime):
    midnight = actual_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = (actual_datetime - midnight).total_seconds()
    return timesincemidnight

def createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date):
    activityTimestamp = line[1]
    activity = []
    activity.append(caseID)
    for feature in line[2:]:
        activity.append(feature)

    #add features: time from trace start, time from last_startdate_event, time from midnight, weekday
    activity.append((activityTimestamp - starttime).total_seconds())
    activity.append((activityTimestamp - lastevtime).total_seconds())
    activity.append(calculateTimeFromMidnight(activityTimestamp))
    activity.append(activityTimestamp.weekday())
    # if there is also end_date add features time from last_enddate_event and event_duration
    if current_activity_end_date is not None:
        activity.append((current_activity_end_date - activityTimestamp).total_seconds())
        # add timestamp end or start to calculate remaining time later
        activity.append(current_activity_end_date)
    else:
        activity.append(activityTimestamp)
    return activity


def move_essential_columns(df, case_id_position, start_date_position):
    columns = df.columns.to_list()
    # move case_id column and start_date column to always know their position
    case = columns[case_id_position]
    start = columns[start_date_position]
    columns.pop(columns.index(case))
    columns.pop(columns.index(start))
    df = df[[case, start] + columns]
    return df

def convert_strings_to_datetime(df, date_format):
    # convert string columns that contain datetime to datetime
    for column in df.columns:
        try:
            #if a number do nothing
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column], format=date_format)
        # exception means it is really a string
        except (ValueError, TypeError, OverflowError):
            pass
    return df

def find_case_finish_time(trace, num_activities):
    # we find the max finishtime for the actual case
    for i in range(num_activities):
        if i == 0:
            finishtime = trace[-i-1][-1]
        else:
            if trace[-i-1][-1] > finishtime:
                finishtime = trace[-i-1][-1]
    return finishtime

def calculate_remaining_time_for_actual_case(traces, num_activities):
    finishtime = find_case_finish_time(traces, num_activities)
    for i in range(num_activities):
        # calculate remaining time to finish the case for every activity in the actual case
        traces[-(i + 1)][-1] = (finishtime - traces[-(i + 1)][-1]).total_seconds()
    return traces

def fill_missing_end_dates(df, start_date_position, end_date_position):
    df[df.columns[end_date_position]] = df.apply(lambda row: row[start_date_position]
                  if row[end_date_position] == 0 else row[end_date_position], axis=1)
    return df

def convert_datetime_columns_to_seconds(df):
    for column in df.columns:
        try:
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column])
            df[column] = (df[column] - pd.to_datetime('1970-01-01 00:00:00')).dt.total_seconds()
        except (ValueError, TypeError, OverflowError):
            pass
    return df

def add_features(df, end_date_position):
    dataset = df.values
    traces = []
    # analyze first dataset line
    caseID = dataset[0][0]
    activityTimestamp = dataset[0][1]
    starttime = activityTimestamp # Start and last are the same
    lastevtime = activityTimestamp
    current_activity_end_date = None
    line = dataset[0]
    if end_date_position is not None:
        # at the begin previous and current end time are the same
        current_activity_end_date = dataset[0][end_date_position]
        line = np.delete(line, end_date_position)
    num_activities = 1
    activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
    traces.append(activity)

    for line in dataset[1:, :]:
        case = line[0]
        if case == caseID:
            # continues the current case
            activityTimestamp = line[1]
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)

            # lasteventtimes become the actual
            lastevtime = activityTimestamp
            traces.append(activity)
            num_activities += 1
        else:
            caseID = case
            traces = calculate_remaining_time_for_actual_case(traces, num_activities)

            activityTimestamp = line[1]
            starttime = activityTimestamp
            lastevtime = activityTimestamp
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
            traces.append(activity)
            num_activities = 1

    # last case
    traces = calculate_remaining_time_for_actual_case(traces, num_activities)
    # construct df again with new features
    columns = df.columns
    if end_date_position is not None:
        columns = columns.delete(end_date_position)
    columns = columns.delete(1)
    columns = columns.to_list()
    if end_date_position is not None:
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday", "event_duration", "remaining_time"])
    else:
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday", "remaining_time"])
    df = pd.DataFrame(traces, columns=columns)
    return df

def sort_df(df):
    df.sort_values([df.columns[0], df.columns[1]], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
    return df

def fillna(df, date_format):
    for i, column in enumerate(df.columns):
        if df[column].dtype != 'object':
            df[column] = df[column].fillna(0)
        else:
            try:
                #datetime columns have null encoded as 0
                pd.to_datetime(df[column], format=date_format)
                df[column] = df[column].fillna(0)
            # exception means it is a string
            except (ValueError, TypeError, OverflowError):
                pass
                #categorical missing values have no sense encoded as 0
                #df[column] = df[column].fillna("Not present")
    return df

def prepare_data_and_add_features(df, case_id_position, start_date_position, date_format, end_date_position):
    df = fillna(df, date_format)
    if end_date_position is not None:
        df = fill_missing_end_dates(df, start_date_position, end_date_position)
    df = convert_strings_to_datetime(df, date_format)
    df = move_essential_columns(df, case_id_position, start_date_position)
    df = sort_df(df)
    df = add_features(df, end_date_position)
    df["weekday"].replace({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}, inplace=True)
    return df