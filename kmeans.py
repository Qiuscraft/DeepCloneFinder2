"""
使用 FAISS 对函数向量进行聚类，并将结果写入 pkl 缓存文件中。
"""

import os
import pickle
import time

import numpy as np

import config
from extract_functions import extract_functions_from_files
from utils.data_class.function_info import FunctionInfo
from utils.data_class.kmeans_result import KMeansResult

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - 运行时导入
    raise RuntimeError("未找到 faiss 模块，请先安装 faiss-cpu 或 faiss-gpu") from exc


def _check_gpu_availability() -> None:
    """检查GPU是否可用"""
    print("(info) [kmeans] Checking GPU availability")
    try:
        ngpu = faiss.get_num_gpus()
        print(f"(info) [kmeans] Detected {ngpu} GPUs")
        if ngpu == 0:
            print("(warning) [kmeans] No GPU detected")
    except Exception as e:
        print(f"(error) [kmeans] Error checking GPU: {e}")

def _kmeans_train(x: np.ndarray, n_centroids: int, n_iter: int, n_redo: int, verbose: bool, use_gpu: bool) -> faiss.Kmeans:
    print(f"(info) [kmeans] Training FAISS KMeans (k={n_centroids}, iter={n_iter})")
    d = x.shape[1]
    
    if use_gpu:
        print("(info) [kmeans] Using GPU")
        try:
            ngpu = faiss.get_num_gpus()
            if ngpu == 0:
                print("(warning) [kmeans] No GPU detected, fallback to CPU")
                use_gpu = False
        except Exception as e:
            print(f"(error) [kmeans] GPU check error: {e}")
            use_gpu = False
    else:
        print("(info) [kmeans] Using CPU")
    
    kmeans = faiss.Kmeans(
        d,
        n_centroids,
        niter=n_iter,
        nredo=n_redo,
        verbose=verbose,
        gpu=use_gpu,
    )
    kmeans.train(x)
    print("(info) [kmeans] Training completed")
    return kmeans

def main() -> None:
    _check_gpu_availability()
    
    start_time = time.time()
    
    print("(info) [kmeans] Loading function embeddings")
    function_embeddings = pickle.load(open(os.path.expanduser(config.emb_cache_filepath), 'rb'))
    vectors = []
    for func in function_embeddings:
        vectors.append(func.embedding)
        
    x = np.vstack(vectors).astype("float32")
    if x.ndim != 2:
        raise ValueError(f"向量数据维度错误，期望2维，实际为 {x.ndim}")
        
    print(f"(info) [kmeans] Vectors matrix shape={x.shape}")
    
    n_centroids = config.kmeans_n_centroids
    if n_centroids <= 0:
        n_centroids = max(1, len(x) // 1024)
        print(f"(info) [kmeans] Adjusted clusters to {n_centroids}")
    elif n_centroids > len(x):
        print(f"(info) [kmeans] Adjusted clusters from {n_centroids} to {len(x)}")
        n_centroids = len(x)
    
    kmeans = _kmeans_train(
        x,
        n_centroids,
        config.kmeans_n_iter,
        config.kmeans_n_redo,
        config.kmeans_verbose,
        config.kmeans_use_gpu
    )
    
    print("(info) [kmeans] Assigning clusters")
    distances, centroids = kmeans.index.search(x, 1)
    print("(info) [kmeans] Assignment completed")
    
    kmeans_result = []

    for i, _ in enumerate(centroids):
        kmeans_result.append(KMeansResult(i, int(centroids[i, 0]), float(distances[i, 0])))
    
    print("(info) [kmeans] Saving cache")
    cache_path = config.kmeans_cache_filepath
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(kmeans_result, f)
    print(f"(info) [kmeans] Saved to {cache_path}")
        
    print(f"(info) [kmeans] Finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
