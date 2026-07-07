import pandas as pd
import pm4py

def find_80_20_trace_split_time(case_study="bpi17_before"):
    file_path = f'case_studies/{case_study}/log_{case_study}.xes'
    print(f"Loading raw event log from {file_path}...")
    
    # Load the raw .xes log using pm4py
    log = pm4py.read_xes(file_path)
    df = pm4py.convert_to_dataframe(log)
    
    # pm4py usually converts timestamps automatically, but we ensure it's a datetime object
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    
    print("Grouping by trace to find case completion timestamps...")
    # Find the END time of each trace using .max()
    trace_ends = df.groupby('case:concept:name')['time:timestamp'].max()
    
    # Calculate the 80th percentile of the trace END times
    split_time = trace_ends.quantile(0.80)
    
    # Format it nicely without the timezone offset for your main script
    formatted_time = split_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*50)
    print(f" RECOMMENDED 80/20 TRACE SPLIT TIME: {formatted_time} ")
    print("="*50 + "\n")
    print(f"Use this command to run your main script:")
    print(f'python data_preprocessing.py --case_study "{case_study}" --split_time "{formatted_time}" --lambda_value 0.5')

if __name__ == "__main__":
    # You can change this to any folder name (e.g., "BPI12", "BAC")
    find_80_20_trace_split_time("bpi17_before")