import os
import json
import pickle
import config
from tqdm import tqdm

def read_prompt_template():
    """读取系统提示和用户提示模板"""
    with open('prompts/system_prompt.md', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    with open('prompts/user_prompt.md', 'r', encoding='utf-8') as f:
        user_prompt_template = f.read()
    
    return system_prompt, user_prompt_template

def create_request_json(centroid_id, function_id, system_prompt, user_prompt):
    """创建请求JSON"""
    return {
        "custom_id": f"request-{centroid_id}-{function_id}",
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": "glm-4-flash",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    }

def main():
    print("=== 代码克隆检测批量请求生成器 ===")
    
    cache_path = None
    if hasattr(config, "clone_class_filepath") and config.clone_class_filepath:
        cache_path = os.path.expanduser(config.clone_class_filepath)

    if not cache_path or not os.path.exists(cache_path):
        print(f"错误: CloneClass 缓存文件 {cache_path} 不存在，请先运行 init_clone_class.py")
        return
        
    print("正在读取提示模板...")
    system_prompt, user_prompt_template = read_prompt_template()
    
    print(f"正在读取 CloneClasses 数据: {cache_path}")
    with open(cache_path, "rb") as f:
        clone_classes = pickle.load(f)
        
    print(f"正在读取 FunctionInfo 数据...")
    with open(os.path.expanduser(config.ef_cache_filepath), "rb") as f:
        functions = pickle.load(f)

    print(f"正在读取 KMeansResult 数据...")
    with open(os.path.expanduser(config.kmeans_cache_filepath), "rb") as f:
        kmeans_results = pickle.load(f)
        
    # 过滤出有效的 CloneClass 且有 suspicious_functions 的
    valid_classes = [cc for cc in clone_classes if cc is not None and cc.suspicious_functions]
    
    if not valid_classes:
        print("没有找到有效的涉及可疑函数的 CloneClass，退出。")
        return

    output_dir = os.path.dirname(config.gbr_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    total_requests = sum(len(cc.suspicious_functions) for cc in valid_classes)
    
    print(f"\n开始生成 JSONL 文件，总请求数: {total_requests}")
    
    output_file = config.gbr_output_file
    f = open(output_file, 'w', encoding='utf-8')
    
    # 将嵌套循环扁平化为生成器，以便 tqdm 能够原生迭代并自动更新进度
    items_to_process = (
        (kmeans_results[cc.representative_function_id].centroid_id, functions[cc.representative_function_id].code_snippet, susp_func_id)
        for cc in valid_classes for susp_func_id in cc.suspicious_functions
    )
    
    for centroid_id, code_snippet_1, susp_func_id in tqdm(items_to_process, total=total_requests, desc="生成请求", unit="请求"):
        code_snippet_2 = functions[susp_func_id].code_snippet
        
        # 替换用户提示模板中的占位符
        user_prompt = user_prompt_template.replace("{{code_snippet_1}}", code_snippet_1)
        user_prompt = user_prompt.replace("{{code_snippet_2}}", code_snippet_2)
        
        # 创建请求JSON
        request_json = create_request_json(centroid_id, susp_func_id, system_prompt, user_prompt)
        
        # 写入JSONL文件
        f.write(json.dumps(request_json, ensure_ascii=False) + '\n')
                
    f.close()
                    
    print(f"\n=== 处理完成 ===")
    print(f"请求已全部保存在文件: {config.gbr_output_file}")
    print(f"总请求数: {total_requests}")

if __name__ == "__main__":
    main()
