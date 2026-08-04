
# import os
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from paretoset import paretoset
# import random
# import tqdm

# from utils.pre_processing_functions import convert_dtypes_bpi12
# from utils.get_features import load_case_study, get_case_study_features
# from utils.transition_system import transition_system
# from utils.recommendation_functions import (
#     act_with_res_func,
#     build_query_instances,
#     next_possible_activities,
#     _to_row_df,
#     nsga2_pareto_search,
#     exhaustive_pareto_search,
#     _build_valid_pairs,
#     _evaluate_candidates
# )

# import warnings
# warnings.filterwarnings("ignore")

# def _default_forbidden_map():
#     bpi17_forbidden = ["O_Accepted"]
#     bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]
#     return {
#         "bpi17_before": bpi17_forbidden,
#         "bpi17_after": bpi17_forbidden,
#         "BPI12": ["O_ACCEPTED"],
#         "BAC": bac_forbidden,
#     }

# def plot_pareto_for_case(
#     case_study: str,
#     method: str,
#     target_case_id: str = None,
#     window_size: int = 5,
#     pop_size: int = 50,
#     n_generations: int = 10,
#     random_state: int = 1234,
#     save_dir: str = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\pareto_front_images"
# ):
#     np.random.seed(random_state)
#     random.seed(random_state)
#     method = method.lower()
    
#     print(f"Loading data for {case_study}...")
#     train_data, test_data, test_log = load_case_study(case_study)

#     if case_study in {"BPI12"}:
#         train_data = convert_dtypes_bpi12(train_data, "experiment")
#         test_data  = convert_dtypes_bpi12(test_data, "experiment")
#         test_log  = convert_dtypes_bpi12(test_log, "experiment")

#     print("Getting features and models...")
#     (
#         predictive_outcome_model,
#         predictive_time_model,
#         case_id_name,
#         activity_column_name,
#         resource_column_name,
#         continuous_features,
#         categorical_features,
#         columns_to_remove,
#     ) = get_case_study_features(case_study)

#     print("Building transition system and maps...")
#     transition_graph, _ = transition_system(
#         train_data,
#         case_id_name=case_id_name,
#         activity_column_name=activity_column_name,
#         window_size=window_size)

#     act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
#     forbidden_map = _default_forbidden_map()
#     forbidden = set(forbidden_map.get(case_study, []))
    
#     query_instances_by_case = build_query_instances(test_data, case_id_name)

#     unique_cases = pd.unique(test_data[case_id_name])
    
#     # MODIFICA: Ricerca del caso con curva ampia e molte soluzioni
#     if target_case_id is None or target_case_id not in unique_cases:
#         print("Ricerca del caso con il maggior numero di soluzioni e la curva di Pareto più distanziata...")
#         best_score = -1
#         best_case_id = None
#         best_info = {}
        
#         for cid in tqdm.tqdm(unique_cases, desc="Valutazione casi"):
#             trace_df = test_log[test_log[case_id_name] == cid]
#             trace_history = trace_df[activity_column_name].tolist()
#             query_instance = _to_row_df(query_instances_by_case[cid])

#             poss = next_possible_activities(trace_history, transition_graph, window_size)
#             poss = [a for a in poss if a not in forbidden]
#             if not poss:
#                 continue

#             valid_pairs = _build_valid_pairs(poss, act_with_res)
#             n_solutions = len(valid_pairs)
            
#             # Filtro: cerchiamo casi con un buon numero di combinazioni possibili
#             if n_solutions < 10:
#                 continue
                
#             # Valutiamo rapidamente le predizioni
#             all_evals = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
#             all_x = all_evals[:, 0]
#             all_y = 1.0 - all_evals[:, 1]
            
#             pareto_vals = np.column_stack((all_x, all_y))
            
#             try:
#                 is_pareto = paretoset(pareto_vals, sense=["max", "max"])
#                 front_x = all_x[is_pareto]
#                 front_y = all_y[is_pareto]
                
#                 # Calcoliamo lo "spread" (ampiezza) della curva: max - min per entrambi gli assi
#                 spread_x = np.max(front_x) - np.min(front_x)
#                 spread_y = np.max(front_y) - np.min(front_y)
                
#                 # Area coperta dal fronte. Più è alta, meno i punti sono "ammassati"
#                 spread_area = spread_x * spread_y
                
#                 # Il nostro criterio: numero di soluzioni MOLTIPLICATO per l'area dello spread
#                 score = n_solutions * spread_area
                
#                 if score > best_score:
#                     best_score = score
#                     best_case_id = cid
#                     best_info = {'n_sol': n_solutions, 'spread_x': spread_x, 'spread_y': spread_y, 'front_size': sum(is_pareto)}
#             except Exception:
#                 continue
                
#         if best_case_id is None:
#             print("Impossibile trovare un caso ideale. Verrà selezionato un caso casuale.")
#             target_case_id = random.choice(unique_cases)
#         else:
#             target_case_id = best_case_id
#             print(f"\nSelezionato automaticamente il case_id: {target_case_id}")
#             print(f"-> Soluzioni totali: {best_info['n_sol']}")
#             print(f"-> Punti sul fronte: {best_info['front_size']}")
#             print(f"-> Spread X (Outcome): {best_info['spread_x']:.3f} | Spread Y (Time): {best_info['spread_y']:.3f}")
#     else:
#         print(f"Utilizzo il case_id fornito: {target_case_id}")

#     # -------- CONTINUA CON LA LOGICA DI PLOT PER IL CASO SCELTO --------
#     trace_df = test_log[test_log[case_id_name] == target_case_id]
#     trace_history = trace_df[activity_column_name].tolist()
#     query_instance = _to_row_df(query_instances_by_case[target_case_id])

#     poss = next_possible_activities(trace_history, transition_graph, window_size)
#     poss = [a for a in poss if a not in forbidden]
    
#     if not poss:
#         print("No possible next activities for this case.")
#         return

#     print("Evaluating complete search space for background points...")
#     valid_pairs = _build_valid_pairs(poss, act_with_res)
#     if not valid_pairs:
#         print("No valid action-resource pairs.")
#         return
        
#     all_evals = _evaluate_candidates(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
    
#     all_x = all_evals[:, 0]
#     all_y = 1.0 - all_evals[:, 1]

#     print(f"Running {method} search...")
#     if method in {"nsga2", "genetic"}:
#         pareto_set = nsga2_pareto_search(
#             query_instance=query_instance,
#             possible_actions=poss,
#             act_with_res=act_with_res,
#             predictive_outcome_model=predictive_outcome_model,
#             predictive_time_model=predictive_time_model,
#             pop_size=pop_size,
#             n_generations=n_generations,
#             random_state=random_state,
#         )
#     elif method == "exhaustive":
#         pareto_set = exhaustive_pareto_search(
#             query_instance,
#             poss,
#             predictive_outcome_model,
#             predictive_time_model,
#             act_with_res,
#         )
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     if not pareto_set:
#         print("Empty Pareto set returned.")
#         return
    
#     front_x_raw = np.array([item[2] for item in pareto_set], dtype=float)
#     front_y_raw = np.array([1.0 - item[3] for item in pareto_set], dtype=float)
    
#     pareto_vals = np.column_stack((front_x_raw, front_y_raw))
#     is_pareto = paretoset(pareto_vals, sense=["max", "max"])
    
#     front_x_sorted = front_x_raw[is_pareto]
#     front_y_sorted = front_y_raw[is_pareto]
    
#     sorted_indices = np.argsort(front_x_sorted)
#     front_x_sorted = front_x_sorted[sorted_indices]
#     front_y_sorted = front_y_sorted[sorted_indices]

#     ideal_point = np.array([1.0, 1.0])
#     front_points = np.column_stack((front_x_sorted, front_y_sorted))
#     distances = np.linalg.norm(front_points - ideal_point, axis=1)
#     best_idx = np.argmin(distances)
#     best_x = front_x_sorted[best_idx]
#     best_y = front_y_sorted[best_idx]

#     plt.figure(figsize=(10, 7))
    
#     plt.scatter(all_x, all_y, color='black', label='Evaluated Pairs (All)', alpha=0.6)
#     plt.scatter(front_x_sorted, front_y_sorted, color='blue', label='Pareto Front', zorder=5)
#     plt.plot(front_x_sorted, front_y_sorted, color='blue', linestyle='--', alpha=0.5)
#     plt.scatter(best_x, best_y, color='red', marker='*', s=200, label='Selected Best Point', zorder=10)
#     plt.scatter(1.0, 1.0, color='green', marker='X', s=150, label='Ideal Point (1,1)', zorder=10)

#     plt.title(f"Pareto Front Analysis\nDataset: {case_study} | Method: {method.upper()} | Case ID: {target_case_id}")
#     plt.xlabel("Predicted Outcome (Maximize)")
#     plt.ylabel("1 - Predicted Time (Maximize)")
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.legend(loc='lower left')
    
#     # Aggiungiamo un margine flessibile in base ai dati effettivi
#     margin_x = (np.max(all_x) - np.min(all_x)) * 0.05
#     margin_y = (np.max(all_y) - np.min(all_y)) * 0.05
    
#     # Assicuriamoci che l'ideale (1,1) sia sempre nel grafico
#     plt.xlim(min(np.min(all_x) - margin_x, -0.05), max(np.max(all_x) + margin_x, 1.05))
#     plt.ylim(min(np.min(all_y) - margin_y, -0.05), max(np.max(all_y) + margin_y, 1.05))

#     os.makedirs(save_dir, exist_ok=True)
#     filename = f"pareto_{case_study}_{method}_{str(target_case_id).replace(':', '_')}.jpg"
#     filepath = os.path.join(save_dir, filename)
#     plt.savefig(filepath, format='jpg', dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Plot saved successfully to {filepath}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Plot Pareto front.')
#     parser.add_argument('--case_study', type=str, required=True, help='Dataset name')
#     parser.add_argument('--method', type=str, required=True, choices=['exhaustive', 'nsga2', 'genetic'])
#     parser.add_argument('--case_id', type=str, default=None)
    
#     try:
#         args = parser.parse_args()
#         plot_pareto_for_case(args.case_study, args.method, args.case_id)
#     except:
#         pass

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
import random
import time
import tqdm

from utils.pre_processing_functions import convert_dtypes_bpi12
from utils.get_features import load_case_study, get_case_study_features
from utils.transition_system import transition_system
from utils.recommendation_functions import (
    act_with_res_func,
    build_query_instances,
    next_possible_activities,
    _to_row_df,
    nsga2_pareto_search,
    exhaustive_pareto_search,
    _build_valid_pairs,
    _evaluate_candidates
)

import warnings
warnings.filterwarnings("ignore")

def _default_forbidden_map():
    bpi17_forbidden = ["O_Accepted"]
    bac_forbidden = ["Network Adjustment Requested", "Back-Office Adjustment Requested"]
    return {
        "bpi17_before": bpi17_forbidden,
        "bpi17_after": bpi17_forbidden,
        "BPI12": ["O_ACCEPTED"],
        "BAC": bac_forbidden,
    }

def evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model):
    """
    Funzione robusta che usa predict_proba se disponibile per l'outcome,
    evitando di avere solo valori secchi 0 o 1.
    """
    base_outcome_row = _to_row_df(query_instance).iloc[0].to_dict()
    base_time_row = _to_row_df(query_instance).iloc[0].to_dict()

    outcome_rows, time_rows = [], []
    for next_act, next_res in valid_pairs:
        o_row = dict(base_outcome_row)
        o_row['NEXT_ACTIVITY'] = next_act
        o_row['NEXT_RESOURCE'] = next_res
        outcome_rows.append(o_row)

        t_row = dict(base_time_row)
        t_row['NEXT_ACTIVITY'] = next_act
        t_row['NEXT_RESOURCE'] = next_res
        time_rows.append(t_row)

    df_out = pd.DataFrame(outcome_rows)
    if hasattr(predictive_outcome_model, "predict_proba"):
        predicted_outcome = predictive_outcome_model.predict_proba(df_out)[:, 1]
    else:
        predicted_outcome = predictive_outcome_model.predict(df_out)

    predicted_total_time = predictive_time_model.predict(pd.DataFrame(time_rows))
    return np.column_stack([predicted_outcome, predicted_total_time])


def run_and_plot_comparison(
    case_study: str,
    target_case_id: str = None,
    window_size: int = 5,
    pop_size: int = 50,
    n_generations: int = 10,
    random_state: int = 1234,
    save_dir: str = "C:\\Users\\Utente\\Desktop\\tesi magistrale\\Code\\multi_obj\\pareto_front_images"
):
    np.random.seed(random_state)
    random.seed(random_state)
    
    print(f"Loading data for {case_study}...")
    train_data, test_data, test_log = load_case_study(case_study)

    if case_study in {"BPI12"}:
        train_data = convert_dtypes_bpi12(train_data, "experiment")
        test_data  = convert_dtypes_bpi12(test_data, "experiment")
        test_log  = convert_dtypes_bpi12(test_log, "experiment")

    print("Getting features and models...")
    (
        predictive_outcome_model,
        predictive_time_model,
        case_id_name,
        activity_column_name,
        resource_column_name,
        continuous_features,
        categorical_features,
        columns_to_remove,
    ) = get_case_study_features(case_study)

    print("Building transition system and maps...")
    transition_graph, _ = transition_system(
        train_data,
        case_id_name=case_id_name,
        activity_column_name=activity_column_name,
        window_size=window_size)

    act_with_res = act_with_res_func(train_data, activity_column_name, resource_column_name)
    forbidden_map = _default_forbidden_map()
    forbidden = set(forbidden_map.get(case_study, []))
    
    query_instances_by_case = build_query_instances(test_data, case_id_name)
    unique_cases = pd.unique(test_data[case_id_name])
    
    # Selezione automatica del caso con la curva più ampia e distribuita
    if target_case_id is None or target_case_id not in unique_cases:
        print("Ricerca del caso con il maggior numero di soluzioni e la curva di Pareto più distanziata...")
        best_score = -1
        best_case_id = None
        
        for cid in tqdm.tqdm(unique_cases, desc="Valutazione casi"):
            trace_df = test_log[test_log[case_id_name] == cid]
            trace_history = trace_df[activity_column_name].tolist()
            query_instance = query_instances_by_case[cid]

            poss = next_possible_activities(trace_history, transition_graph, window_size)
            poss = [a for a in poss if a not in forbidden]
            if not poss:
                continue

            valid_pairs = _build_valid_pairs(poss, act_with_res)
            n_solutions = len(valid_pairs)
            if n_solutions < 10:
                continue
                
            all_evals = evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
            all_x = all_evals[:, 0]
            all_y = 1.0 - all_evals[:, 1]
            
            pareto_vals = np.column_stack((all_x, all_y))
            try:
                is_pareto = paretoset(pareto_vals, sense=["max", "max"])
                front_x = all_x[is_pareto]
                front_y = all_y[is_pareto]
                
                spread_x = np.max(front_x) - np.min(front_x)
                spread_y = np.max(front_y) - np.min(front_y)
                spread_area = spread_x * spread_y
                score = n_solutions * spread_area
                
                if score > best_score:
                    best_score = score
                    best_case_id = cid
            except Exception:
                continue
                
        if best_case_id is None:
            target_case_id = random.choice(unique_cases)
        else:
            target_case_id = best_case_id
        print(f"\nCaso selezionato automaticamente: {target_case_id}")
    else:
        print(f"Utilizzo il case_id fornito: {target_case_id}")

    # Estrazione dati per il caso scelto
    trace_df = test_log[test_log[case_id_name] == target_case_id]
    trace_history = trace_df[activity_column_name].tolist()
    query_instance = query_instances_by_case[target_case_id]

    poss = next_possible_activities(trace_history, transition_graph, window_size)
    poss = [a for a in poss if a not in forbidden]
    
    if not poss:
        print("Nessuna attività successiva possibile per questo caso.")
        return

    valid_pairs = _build_valid_pairs(poss, act_with_res)
    if not valid_pairs:
        print("Nessuna coppia azione-risorsa valida.")
        return

    # =========================================================================
    metodi = ["exhaustive", "nsga2"]
    
    for method in metodi:
        print(f"\nEsecuzione metodo: {method.upper()}...")
        t_start = time.time()
        
        if method == "exhaustive":
            # Tempo totale per la computazione di tutti i punti dello spazio di ricerca
            all_evals = evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
            pareto_set = exhaustive_pareto_search(query_instance, poss, predictive_outcome_model, predictive_time_model, act_with_res)
            elapsed_time = time.time() - t_start
            
            all_x = all_evals[:, 0]
            all_y = 1.0 - all_evals[:, 1]
            
        else:  # nsga2
            # Tempo impiegato dall'algoritmo genetico per trovare il fronte
            all_evals = evaluate_robust(valid_pairs, query_instance, predictive_outcome_model, predictive_time_model)
            all_x = all_evals[:, 0]
            all_y = 1.0 - all_evals[:, 1]
            
            pareto_set = nsga2_pareto_search(
                query_instance=query_instance,
                possible_actions=poss,
                act_with_res=act_with_res,
                predictive_outcome_model=predictive_outcome_model,
                predictive_time_model=predictive_time_model,
                pop_size=pop_size,
                n_generations=n_generations,
                random_state=random_state,
            )
            elapsed_time = time.time() - t_start

        if not pareto_set:
            print(f"Set di Pareto vuoto per {method}. Salto il plot.")
            continue
        
        front_x_raw = np.array([item[2] for item in pareto_set], dtype=float)
        front_y_raw = np.array([1.0 - item[3] for item in pareto_set], dtype=float)
        
        pareto_vals = np.column_stack((front_x_raw, front_y_raw))
        is_pareto = paretoset(pareto_vals, sense=["max", "max"])
        
        front_x_sorted = front_x_raw[is_pareto]
        front_y_sorted = front_y_raw[is_pareto]
        
        sorted_indices = np.argsort(front_x_sorted)
        front_x_sorted = front_x_sorted[sorted_indices]
        front_y_sorted = front_y_sorted[sorted_indices]

        ideal_point = np.array([1.0, 1.0])
        front_points = np.column_stack((front_x_sorted, front_y_sorted))
        distances = np.linalg.norm(front_points - ideal_point, axis=1)
        best_idx = np.argmin(distances)
        best_x = front_x_sorted[best_idx]
        best_y = front_y_sorted[best_idx]

        # Generazione Grafico
        plt.figure(figsize=(10, 7))
        
        plt.scatter(all_x, all_y, color='black', label='Evaluated Pairs (All)', alpha=0.6)
        plt.scatter(front_x_sorted, front_y_sorted, color='blue', label='Pareto Front', zorder=5)
        plt.plot(front_x_sorted, front_y_sorted, color='blue', linestyle='--', alpha=0.5)
        plt.scatter(best_x, best_y, color='red', marker='*', s=200, label='Selected Best Point', zorder=10)
        plt.scatter(1.0, 1.0, color='green', marker='X', s=150, label='Ideal Point (1,1)', zorder=10)

        title_text = (
            f"Pareto Front Analysis ({method.upper()})\n"
            f"Dataset: {case_study} | Case ID: {target_case_id}\n"
            f"Execution Time: {elapsed_time:.4f} seconds"
        )
        plt.title(title_text)
        plt.xlabel("Predicted Outcome (Probability Maximize)")
        plt.ylabel("1 - Predicted Time (Maximize)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='lower left')
        
        margin_x = (np.max(all_x) - np.min(all_x)) * 0.05 if np.max(all_x) != np.min(all_x) else 0.05
        margin_y = (np.max(all_y) - np.min(all_y)) * 0.05 if np.max(all_y) != np.min(all_y) else 0.05
        
        plt.xlim(min(np.min(all_x) - margin_x, -0.05), max(np.max(all_x) + margin_x, 1.05))
        plt.ylim(min(np.min(all_y) - margin_y, -0.05), max(np.max(all_y) + margin_y, 1.05))

        os.makedirs(save_dir, exist_ok=True)
        filename = f"pareto_{case_study}_{method}_{str(target_case_id).replace(':', '_')}.jpg"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grafico salvato in: {filepath} (Tempo: {elapsed_time:.4f}s)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Pareto comparison with execution times.')
    parser.add_argument('--case_study', type=str, required=True, help='Dataset name')
    parser.add_argument('--case_id', type=str, default=None, help='Specific case ID (optional)')
    
    try:
        args = parser.parse_args()
        run_and_plot_comparison(case_study=args.case_study, target_case_id=args.case_id)
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")