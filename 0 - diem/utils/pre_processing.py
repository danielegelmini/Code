import pandas as pd
import numpy as np
import pm4py
from utils.pre_processing_functions import prepare_data_and_add_features, add_next_act_res, preprocessing_activity_frequency, getting_total_time
from dateutil import parser



def data_pre_processing(path_to_df, case_id_position, start_date_position, date_format, end_date_position, case_id_name, 
                        start_date_name, end_date_name, activity_column_name, resource_column_name):
    print("Loading data...")
    data = pd.read_csv(path_to_df, parse_dates=[start_date_name, end_date_name], dayfirst=True)
    
    # Function "format_dataframe" creates another column for start timestamp, case index, and event index
    df = pm4py.utils.format_dataframe(data,  case_id =  case_id_name, activity_key = activity_column_name, timestamp_key = end_date_name, start_timestamp_key = start_date_name) 
    df["time_timestamp"] = df[end_date_name] # Avoiding losing information when preprocessing
    df = df.drop(columns=["@@index", "@@case_index"], axis = 1) # Dropping columns that repeat information

    print("Start pre-processing data...")

    print("...adding total time...")
    df = getting_total_time(df, case_id_name, start_date_name, end_date_name)

    print("...adding activity frequency...")
    df = preprocessing_activity_frequency(df, activity_column_name, case_id_name, start_date_name)
    
    print("...adding next activity and next resource...")
    df = add_next_act_res(df, activity_column_name, resource_column_name, case_id_name)
    # # Combining Next Activity and Next Resource to form Next Action
    # df['Next_Action'] = list(zip(df['NEXT_ACTIVITY'], df['NEXT_RESOURCE']))
    # df = df.drop(columns=["NEXT_ACTIVITY", "NEXT_RESOURCE"], axis = 1)

    print("...adding features...")
    df = prepare_data_and_add_features(df, case_id_position, start_date_position, 
                                       date_format, end_date_position)

    df = df.rename(columns={"time_timestamp": "time:timestamp", "start_timestamp": "start:timestamp", "leadtime":"total_time"})
    print("Finished pre-processing!")
    print("Saving pre-processed data...")
    df.to_csv("./data/preprocessed_data.csv", index=False)
    return df
