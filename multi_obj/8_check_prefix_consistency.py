#!/usr/bin/env python3
"""
8_check_prefix_consistency.py

Strumento di validazione: controlla che le simulazioni prodotte per i vari metodi
contengano esattamente il prefisso storico inalterato all'inizio di ogni traccia.

Uso:
    python 8_check_prefix_consistency.py --case_study BPI12 --n_sim 10
"""

import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Verifica che i prefissi simulati corrispondano a quelli reali.")
    parser.add_argument("--case_study", type=str, required=True, help="Nome del caso studio sotto case_studies/")
    parser.add_argument("--base_dir", type=str, default=".", help="Directory base")
    parser.add_argument("--n_sim", type=int, default=10, help="Numero di file di simulazione da controllare")
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    case_dir = base_dir / "case_studies" / args.case_study
    test_log_path = case_dir / "test_log.csv"

    if not test_log_path.exists():
        print(f"Errore: {test_log_path} non trovato.")
        return

    print(f"Caricamento log dei prefissi originali: {test_log_path}")
    test_log = pd.read_csv(test_log_path, dtype={"case:concept:name": str})
    
    # Assicuriamo l'ordinamento cronologico
    test_log['time:timestamp'] = pd.to_datetime(test_log['time:timestamp'], format="mixed", utc=True, errors="coerce")
    test_log = test_log.sort_values(by=["case:concept:name", "time:timestamp"])

    # Dizionario con i dati del prefisso reale per ogni caso
    prefix_info = {}
    for case_id, group in test_log.groupby("case:concept:name"):
        prefix_info[case_id] = {
            "length": len(group),
            "activities": group["concept:name"].tolist(),
            "resources": group["org:resource"].astype(str).tolist()
        }
        
    print(f"Estratti {len(prefix_info)} prefissi unici da test_log.csv.\n")

    methods_to_check = ["baseline", "exhaustive", "nsga2"]
    
    for method in methods_to_check:
        print(f"=== Controllo Metodo: {method.upper()} ===")
        sim_dir = case_dir / "prosit_simulation_results" / method
        
        if not sim_dir.exists():
            print(f"  [!] Cartella {sim_dir} non trovata. Salto.\n")
            continue
            
        for i in range(1, args.n_sim + 1):
            sim_file = sim_dir / f"sim_{i}.csv"
            if not sim_file.exists():
                print(f"  [!] File {sim_file.name} non trovato.")
                continue
                
            sim_df = pd.read_csv(sim_file, dtype={"case:concept:name": str})
            sim_df['time:timestamp'] = pd.to_datetime(sim_df['time:timestamp'], format="mixed", utc=True, errors="coerce")
            sim_df = sim_df.sort_values(by=["case:concept:name", "time:timestamp"])
            
            mismatches = 0
            missing_cases = 0
            
            for case_id, p_info in prefix_info.items():
                sim_case = sim_df[sim_df["case:concept:name"] == case_id]
                if sim_case.empty:
                    missing_cases += 1
                    continue
                    
                # Estraiamo l'inizio della traccia simulata per una lunghezza pari a quella del prefisso
                sim_prefix_activities = sim_case.head(p_info["length"])["concept:name"].tolist()
                
                if sim_prefix_activities != p_info["activities"]:
                    mismatches += 1
                    # Stampiamo i dettagli del primo errore trovato per debug
                    if mismatches == 1:
                        print(f"\n    -> ESEMPIO DISCREPANZA SUL CASO {case_id} ({sim_file.name}):")
                        print(f"       Prefisso Atteso : {p_info['activities']}")
                        print(f"       Prefisso Reale  : {sim_prefix_activities}\n")
                    
            total = len(prefix_info)
            matched = total - mismatches - missing_cases
            
            if mismatches == 0 and missing_cases == 0:
                print(f"  [OK] {sim_file.name}: {matched}/{total} prefissi corrispondono perfettamente.")
            else:
                print(f"  [ERRORE] {sim_file.name}: Solo {matched}/{total} corretti. "
                      f"({mismatches} errati, {missing_cases} mancanti)")
        print()

if __name__ == "__main__":
    main()
