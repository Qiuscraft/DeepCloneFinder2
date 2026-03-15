import os
import pickle
import random
import config
from typing import List, Dict, Optional

from extract_functions import extract_functions_from_files
from utils.clone_class.clone_class import CloneClass
from utils.data_class.function_info import FunctionInfo
from utils.data_class.kmeans_result import KMeansResult

def init_clone_classes() -> List[Optional[CloneClass]]:
    functions = extract_functions_from_files()
    kmeans_results = pickle.load(open(os.path.expanduser(config.kmeans_cache_filepath), "rb"))

    # 构建 id 到 FunctionInfo 以及 id 到距离的映射
    func_map = {f.id: f for f in functions}
    
    # 找出最大 centroid_id
    max_centroid = max(kr.centroid_id for kr in kmeans_results)
    
    # 按照 centroid 分组函数 ID，并记录聚类结果
    kr_by_centroid: Dict[int, List[KMeansResult]] = {i: [] for i in range(max_centroid + 1)}
    kr_map: Dict[int, KMeansResult] = {}
    for kr in kmeans_results:
        kr_by_centroid[kr.centroid_id].append(kr)
        kr_map[kr.id] = kr
        
    clone_classes: List[Optional[CloneClass]] = [None] * (max_centroid + 1)

    for centroid in range(max_centroid + 1):
        krs = kr_by_centroid[centroid]
        if not krs:
            continue
            
        # 先过滤掉不在 func_map 中的（避免数据不一致）
        valid_krs = [kr for kr in krs if kr.id in func_map]
        if not valid_krs:
            continue

        # 找出 distance 最小的函数对应的 KMeansResult
        min_distance = min(kr.distance for kr in valid_krs)
        
        # 将相同的函数放到 confirmed_functions，剩下的保留在 suspicious_functions
        confirmed_krs = [kr for kr in valid_krs if kr.distance == min_distance]
        suspicious_krs = [kr for kr in valid_krs if kr.distance != min_distance]
        
        # 将 distance 最小的一个函数作为代表函数
        representative_id = confirmed_krs[0].id
        
        # 初始化 CloneClass
        cc = CloneClass(representative_id)
        cc.confirmed_functions = [kr.id for kr in confirmed_krs]
        cc.suspicious_functions = [kr.id for kr in suspicious_krs]
        
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
        rep_id = random_cc.representative_function_id
        print(f"(info) Randomly selected CloneClass:")
        print(f"  - Representative Function ID: {rep_id}")
        print(f"  - Confirmed functions count: {len(random_cc.confirmed_functions)}")
        for f_id in random_cc.confirmed_functions[:5]:  # print up to 5
            print(f"      * Function ID: {f_id}")
        if len(random_cc.confirmed_functions) > 5:
            print("      * ...")
        print(f"  - Suspicious functions count: {len(random_cc.suspicious_functions)}")
    else:
        print("(info) No valid CloneClass found.")
