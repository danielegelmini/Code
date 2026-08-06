# give a fil path i want to read it and understand how many cases it has. the file is in .xes and the first column is the case id. i want to know how many unique case ids are in the file.

file_path_4 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_after\\test_log.csv"
file_path_3 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_before\\test_log.csv"
file_path_2 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BPI12\\test_log.csv"
file_path_1 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BAC\\test_log.csv"

file_path_5 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BAC\\prosit_simulation_results\\baseline\\sim_1.csv"
file_path_6 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BAC\\prosit_simulation_results\\exhaustive\\sim_1.csv"
file_path_7 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BAC\\prosit_simulation_results\\nsga2\\sim_1.csv"

file_path_8 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BPI12\\prosit_simulation_results\\baseline\\sim_1.csv"
file_path_9 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BPI12\\prosit_simulation_results\\exhaustive\\sim_1.csv"
file_path_10 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\BPI12\\prosit_simulation_results\\nsga2\\sim_1.csv"

file_path_11 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_before\\prosit_simulation_results\\baseline\\sim_1.csv"
file_path_12 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_before\\prosit_simulation_results\\exhaustive\\sim_1.csv"
file_path_13 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_before\\prosit_simulation_results\\nsga2\\sim_1.csv"

file_path_14 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_after\\prosit_simulation_results\\baseline\\sim_1.csv"
file_path_15 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_after\\prosit_simulation_results\\exhaustive\\sim_1.csv"
file_path_16 = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\case_studies\\bpi17_after\\prosit_simulation_results\\nsga2\\sim_1.csv"



import pandas as pd

def print_result(dataset_name, original_file, files):

    df_original = pd.read_csv(original_file)
    unique_case_ids = df_original.iloc[:, 0].nunique()
    print(f"The log dataset {dataset_name} has {unique_case_ids} events")

    simulation_method = ["baseline", "exhaustive", "nsga2"]
    
    for file_path, method in zip(files,simulation_method):
        df = pd.read_csv(file_path)
        unique_case_ids = df.iloc[:, 0].nunique()
        print(f"The method {method} has {unique_case_ids} events")



print_result("BAC", file_path_1, [file_path_5, file_path_6, file_path_7])

#print_result("BPI12", file_path_2, [file_path_8, file_path_9, file_path_10])

print_result("bpi17_before", file_path_3, [file_path_11, file_path_12, file_path_13])

print_result("bpi17_after", file_path_4, [file_path_14, file_path_15, file_path_16])

    