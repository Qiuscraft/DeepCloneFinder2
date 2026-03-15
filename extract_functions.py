import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List
import config
from utils.data_class.function_info import FunctionInfo
from utils.java_code.java_parser import JavaParser


def get_java_files_content() -> dict:
    """
    获取配置中的 ef_dataset_path，深度遍历所有 .java 文件，
    并并发读取它们的内容，返回 Dict[相对路径, 文件内容]。
    """
    # 处理 ~ 等路径符号
    dataset_path = os.path.expanduser(config.ef_dataset_path)
    
    java_files = []
    # 深度遍历所有的.java文件
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.java'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_path)
                java_files.append((full_path, rel_path))
                
    results = {}
    
    def read_file(paths):
        full_path, rel_path = paths
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return rel_path, content
        except Exception as e:
            print(f"(error) [extract_functions] Error reading {full_path}: {e}")
            return rel_path, None

    # 获取配置的 workers 数量
    max_workers = config.ef_read_max_workers
    
    # 采用多线程 (ThreadPoolExecutor) 来并发读取文件内容，提升效率
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rel_path, content in executor.map(read_file, java_files):
            if content is not None:
                results[rel_path] = content
                
    return results


def _parse_single_file(args: tuple) -> List[FunctionInfo]:
    rel_path, content = args
    if not content or not content.strip():
        return []
        
    try:
        parser = JavaParser(file_path=rel_path, source_code=content)
        return parser.extract_functions()
    except Exception as e:
        print(f"(error) [extract_functions] Error parsing {rel_path}: {e}")
        return []

def extract_functions_from_files() -> List[FunctionInfo]:
    """
    如果cache存在，直接读取cache，如果不存在，再读取 files_content 并解析。
    由于使用 javalang 构建和解析语法树属于典型的 CPU 密集型任务，
    且在 Python 中受 GIL 限制，多线程在这个场景下无法实现真正的并行计算。
    因此，必须使用多进程 (ProcessPoolExecutor) 才能充分利用多核 CPU 来加速。
    """
    cache_path = None
    if hasattr(config, 'ef_cache_filepath') and config.ef_cache_filepath:
        cache_path = os.path.expanduser(config.ef_cache_filepath)
        if os.path.exists(cache_path):
            print(f"(info) [extract_functions] Loading extracted functions from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    files_content = get_java_files_content()
    
    results: List[FunctionInfo] = []
    
    # 尽可能使用可用的 CPU 核心进行多进程解析
    max_workers = config.ef_parse_max_workers
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for funcs in executor.map(_parse_single_file, files_content.items()):
            results.extend(funcs)

    i = 0    
    for func in results:
        func.id = i
        i += 1

    if cache_path:
        print(f"(info) [extract_functions] Saving extracted functions to cache: {cache_path}")
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    return results


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    function_infos = extract_functions_from_files()
    print(f"(info) [extract_functions] Total functions extracted: {len(function_infos)}")
    print(f"(info) [extract_functions] Time taken to extract functions: {time.time() - start_time:.2f} seconds")

    # 随机显示1个函数的 FunctionInfo
    import random
    func = random.choice(function_infos)
    print(f"\n=== Sample FunctionInfo ===")
    print(f"\nFunction ID: {func.id}")
    print(f"File Path: {func.path}")
    print(f"Lines: {func.start_line} - {func.end_line}")
    print(f"Code Snippet: \n{func.code_snippet}")
