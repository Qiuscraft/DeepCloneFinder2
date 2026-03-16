import os
import pickle
import csv
from itertools import combinations
from tqdm import tqdm
import config
from utils.clone_class.clone_class import CloneClass
from utils.data_class.function_info import FunctionInfo

def main():
    print(f"(info) [generate_final_csv] Loading function info from {config.ef_cache_filepath}...")
    with open(config.ef_cache_filepath, "rb") as f:
        function_infos = pickle.load(f)
        
    print(f"(info) [generate_final_csv] Loading clone classes from {config.icc_cache_filepath}...")
    with open(config.icc_cache_filepath, "rb") as f:
        clone_classes = pickle.load(f)
        
    print(f"(info) [generate_final_csv] Generating clone pairs to {config.gfc_output_csv}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.gfc_output_csv), exist_ok=True)
    
    with open(config.gfc_output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        pair_count = 0
        for cc in tqdm(clone_classes, desc="(info) [generate_final_csv] Generating clone pairs"):
            # Combine representative_function_id and confirmed_functions
            # Remove duplicates just in case
            group_ids = list(set([cc.representative_function_id] + cc.confirmed_functions))
            
            # Form all unique pairs from the group
            for id1, id2 in combinations(group_ids, 2):
                if 0 <= id1 < len(function_infos) and 0 <= id2 < len(function_infos):
                    info1 = function_infos[id1]
                    info2 = function_infos[id2]
                    
                    sub1 = os.path.basename(os.path.dirname(info1.path))
                    name1 = os.path.basename(info1.path)
                    
                    sub2 = os.path.basename(os.path.dirname(info2.path))
                    name2 = os.path.basename(info2.path)
                    
                    writer.writerow([sub1, name1, info1.start_line, info1.end_line,
                                     sub2, name2, info2.start_line, info2.end_line])
                    pair_count += 1
                    
        print(f"(info) [generate_final_csv] Done! Generated {pair_count} clone pairs in {config.gfc_output_csv}.")

if __name__ == "__main__":
    main()
