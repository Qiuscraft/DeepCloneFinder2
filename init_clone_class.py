import os
import pickle
import random
import config
from typing import List, Dict, Optional

from extract_functions import extract_functions_from_files
from utils.clone_class.clone_class import CloneClass
from utils.java_code.function_info import FunctionInfo

def init_clone_classes() -> List[Optional[CloneClass]]:
    functions = extract_functions_from_files()
    if not functions:
        return []

    # 找出最大 centroid
    max_centroid = max(f.centroid for f in functions)
    
    # 按照 centroid 分组函数
    funcs_by_centroid: Dict[int, List[FunctionInfo]] = {i: [] for i in range(max_centroid + 1)}
    for f in functions:
        funcs_by_centroid[f.centroid].append(f)
        
    clone_classes: List[Optional[CloneClass]] = [None] * (max_centroid + 1)

    for centroid in range(max_centroid + 1):
        f_list = funcs_by_centroid[centroid]
        if not f_list:
            continue
            
        # 找出 distance 最小的函数
        min_distance = min(f.distance for f in f_list)
        
        # 将相同的函数放到 confirmed_functions，剩下的保留在 suspicious_functions
        confirmed = [f for f in f_list if f.distance == min_distance]
        suspicious = [f for f in f_list if f.distance != min_distance]
        
        # 将 distance 最小的一个函数作为代表函数
        representative = confirmed[0]
        
        # 初始化 CloneClass
        cc = CloneClass(representative)
        cc.confirmed_functions = confirmed
        cc.suspicious_functions = suspicious
        
        clone_classes[centroid] = cc

    return clone_classes

if __name__ == "__main__":
    cache_path = None
    if hasattr(config, "icc_cache_filepath") and config.icc_cache_filepath:
        cache_path = os.path.expanduser(config.icc_cache_filepath)

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            clone_classes = pickle.load(f)
        print(f"(info) [init_clone_classes] Data loaded from {cache_path}")
    else:
        clone_classes = init_clone_classes()
        
        # 持久化到 cache
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(clone_classes, f)
            print(f"(info) [init_clone_classes] Data successfully saved to {cache_path}")
        else:
            print(f"(error) [init_clone_classes] Error: icc_cache_filepath not found in config.")

    valid_classes = [cc for cc in clone_classes if cc is not None]
    if valid_classes:
        random_cc = random.choice(valid_classes)
        rep = random_cc.representative_function
        print(f"(info) Randomly selected CloneClass (Centroid {rep.centroid}):")
        print(f"  - Representative: {rep.path} (lines {rep.start_line}-{rep.end_line})")
        print(f"  - Confirmed functions count: {len(random_cc.confirmed_functions)}")
        for f in random_cc.confirmed_functions[:5]:  # print up to 5
            print(f"      * {f.path} (lines {f.start_line}-{f.end_line}, dist: {f.distance:.4f})")
        if len(random_cc.confirmed_functions) > 5:
            print("      * ...")
        print(f"  - Suspicious functions count: {len(random_cc.suspicious_functions)}")
    else:
        print("(info) No valid CloneClass found.")
