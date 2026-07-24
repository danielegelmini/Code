import pandas as pd
import numpy as np
from collections import Counter
from dateutil import parser

def get_activities(df, id, case_id_name, activity_column_name):
    # Return list of activities for each trace
    trace_df = df[df[case_id_name] == id]
    list_activities = trace_df[activity_column_name] 
    return list_activities

def get_resources(df, id, activity, activity_column_name, resource_column_name, case_id_name):
    # Extract resource or list of resources perform a certain activity
    trace_df = df[df[case_id_name] == id]
    res = trace_df[trace_df[activity_column_name] == activity][resource_column_name].tolist()
    return res

def act_with_res(df, activity_column_name, resource_column_name):
    """
    Generates a dictionary mapping each unique activity to a list of unique resources associated with it.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.
        
    Returns:
        dict: A dictionary where keys are unique activities and values are lists of unique resources associated with each activity.
    """
    
    list_activities = df[activity_column_name].unique()
    result = {}
    for act in list_activities:
        res = df[df[activity_column_name] == act][resource_column_name].unique().tolist()
        if "missing" in res:
            res.remove("missing")
        result[act] = res
    return result

def res_freq(df, lst_activities, list_act_res, activity_column_name, resource_column_name):
    """
    Calculate the frequency of resources performing a list of activities.

    Parameters:
        df (pd.DataFrame): The dataframe containing the activity and resource data.
        lst_activities (list): A list of activities to consider.
        list_act_res (dict): A dictionary where keys are activities and values are lists of resources.
        activity_column_name (str): The name of the column in the dataframe that contains activity names.
        resource_column_name (str): The name of the column in the dataframe that contains resource names.

    Returns:
        dict: A nested dictionary where the keys are activities and the values are dictionaries.
              These inner dictionaries have resources as keys and their corresponding frequencies as values,
              sorted in descending order of frequency.
    """
    res_freq = {}
    for act in lst_activities:
        lst_res = list_act_res[act]
        res_freq[act] = {}
        for res in lst_res:
            freq = df[(df[activity_column_name] == act) & (df[resource_column_name] == res)].shape[0]
            res_freq[act][res] = freq
        res_freq[act] = dict(sorted(res_freq[act].items(), key=lambda item: -item[1])) # Sorted with descending order
    return res_freq
