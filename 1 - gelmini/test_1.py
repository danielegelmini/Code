import pandas as pd
import pm4py

# 1. Load the XES file
file_path = r"C:\Users\Utente\Desktop\tesi magistrale\Code\case_studies\BAC\log_BAC.xes"
print(f"Loading {file_path}...")
log = pm4py.read_xes(file_path)

# 2. Convert to a Pandas DataFrame for easier exploration
df = pm4py.convert_to_dataframe(log)

# 3. Basic DataFrame properties
print("\n--- BASIC DATASET PROPERTIES ---")
print(f"Total number of events (rows): {len(df)}")
print(f"Total number of columns (features): {len(df.columns)}")
print("\nColumns available:")
for col in df.columns:
    print(f" - {col}")

# 4. Process Mining Specific Properties
# In XES, 'case:concept:name' is the Case ID, and 'concept:name' is the Activity
case_id_col = 'case:concept:name'
activity_col = 'concept:name'
timestamp_col = 'time:timestamp'

print("\n--- PROCESS MINING PROPERTIES ---")
num_cases = df[case_id_col].nunique()
num_activities = df[activity_col].nunique()
print(f"Number of unique cases (process instances): {num_cases}")
print(f"Number of unique activities: {num_activities}")

print("\nTop 5 most frequent activities:")
print(df[activity_col].value_counts().head())

# Timeframe of the log
min_time = df[timestamp_col].min()
max_time = df[timestamp_col].max()
print(f"\nLog starts at: {min_time}")
print(f"Log ends at: {max_time}")

# 5. Trace length analysis (events per case)
trace_lengths = df.groupby(case_id_col).size()
print("\n--- TRACE LENGTH (EVENTS PER CASE) ---")
print(f"Minimum events in a case: {trace_lengths.min()}")
print(f"Maximum events in a case: {trace_lengths.max()}")
print(f"Average events per case: {trace_lengths.mean():.2f}")

# 6. Look at the first few rows of a single specific case
first_case_id = df[case_id_col].iloc[0]
print(f"\n--- SAMPLE TRACE (Case ID: {first_case_id}) ---")
sample_case = df[df[case_id_col] == first_case_id].sort_values(timestamp_col)
print(sample_case[[case_id_col, activity_col, timestamp_col, 'org:resource']].head(10))