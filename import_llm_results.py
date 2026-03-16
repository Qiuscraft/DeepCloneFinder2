import json
import pickle
import re
import tqdm

import config


def extract_llm_fields(input_str):
    """
    从给定的字符串输入中提取model、centroid_id、function_id、reasoning和clone_type字段
    
    Args:
        input_str: 包含LLM结果的JSON字符串
    
    Returns:
        包含提取字段的字典
    
    Raises:
        ValueError: 当所有解析策略都失败时
    """
    # 解析整个输入为JSON
    data = json.loads(input_str)
    
    # 提取model
    model = data['response']['body']['model']
    
    # 提取custom_id并分割为centroid_id和function_id
    custom_id = data['custom_id']
    parts = custom_id.split('-')
    centroid_id = parts[1]
    function_id = parts[2]
    
    # 提取content
    content = data['response']['body']['choices'][0]['message']['content']
    
    # 初始化reasoning和clone_type
    reasoning = None
    clone_type = None
    
    # 策略1: 去除前导和尾随空格，删除```json和```，按JSON格式读取
    try:
        trimmed_content = content.strip()
        # 处理代码块标记
        if trimmed_content.startswith('```json'):
            json_content = trimmed_content[7:-3].strip()
        elif trimmed_content.startswith('```'):
            json_content = trimmed_content[3:-3].strip()
        else:
            json_content = trimmed_content
        # 解析为JSON
        content_data = json.loads(json_content)
        reasoning = content_data['reasoning']
        clone_type = content_data['clone_type']
    except Exception:
        # 策略1失败
        pass
    
    # 检查策略1是否成功
    if reasoning and clone_type:
        try:
            centroid_id = int(centroid_id)
            function_id = int(function_id)
            clone_type = int(clone_type)
        except Exception as e:
            raise ValueError("(error) [import_llm_results] centroid_id, function_id, clone_type must be INT") from e
        return {
            'model': model,
            'centroid_id': centroid_id,
            'function_id': function_id,
            'reasoning': reasoning,
            'clone_type': clone_type
        }

    # 策略2: 使用正则表达式匹配
    try:
        # 匹配reasoning字段（支持带引号和不带引号的键）
        reasoning_pattern = re.compile(r'(?:"reasoning"|reasoning)\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        reasoning_match = reasoning_pattern.search(content)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
            # 处理转义的引号
            reasoning = reasoning.replace('\\"', '"')
        
        # 匹配clone_type字段（支持带引号和不带引号的键，支持带引号的字符串和无引号的数字）
        clone_type_pattern = re.compile(r'(?:"clone_type"|clone_type)\s*:\s*(?:"([^"]*)"|(\d+))')
        clone_type_match = clone_type_pattern.search(content)
        if clone_type_match:
            if clone_type_match.group(1):
                clone_type = clone_type_match.group(1)
            else:
                clone_type = clone_type_match.group(2)
    except Exception as e:
        # 策略2失败
        raise ValueError("(error) [import_llm_results] Failed to parse reasoning and clone_type from content") from e

    # 检查策略2是否成功
    if reasoning and clone_type:
        try:
            centroid_id = int(centroid_id)
            function_id = int(function_id)
            clone_type = int(clone_type)
        except Exception as e:
            raise ValueError("(error) [import_llm_results] centroid_id, function_id, clone_type must be INT") from e
        return {
            'model': model,
            'centroid_id': centroid_id,
            'function_id': function_id,
            'reasoning': reasoning,
            'clone_type': clone_type
        }
    
    # 两种策略都失败，抛出错误
    raise ValueError("(error) [import_llm_results] Failed to parse reasoning and clone_type from content")

def process_llm_results_file(file_path):
    """
    处理包含LLM结果的JSONL文件，对每一行调用extract_llm_fields函数
    
    Args:
        file_path: JSONL文件的路径
    
    Returns:
        包含所有解析结果的列表
    """
    results = []
    failures = []
    total_count = 0
    
    # 计算文件总行数，用于tqdm进度条
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    # 读取文件并处理每一行
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, total=line_count, desc="Processing LLM results"):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                # 调用extract_llm_fields函数解析当前行
                parsed_result = extract_llm_fields(line)
                results.append(parsed_result)
            except Exception as e:
                # 解析失败，记录失败信息
                failures.append(line)
    
    # 输出失败信息
    failure_count = len(failures)
    print(f"(info) [import_llm_results] Processing completed. Total records: {total_count}, Failed records: {failure_count}.")
    
    if failures:
        print("\n(warn) [import_llm_results] Failed records:")
        for i, failed_line in enumerate(failures[:10], 1):  # 只显示前10个失败的记录
            print(f"{i}. {failed_line}")
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failed records")
    
    return results

def update_clone_classes(results, filepath=config.icc_cache_filepath):
    """
    根据给定的解析结果更新克隆类列表。

    Args:
        results: 包含解析后LLM结果的列表
        filepath: 克隆类列表的缓存文件路径，默认为config.icc_cache_filepath

    Returns:
        更新后的克隆类列表
    """
    print(f"(info) [import_llm_results] Load clone classes from {filepath}...")
    with open(filepath, 'rb') as f:
        clone_classes = pickle.load(f)
        
    for item in tqdm.tqdm(results, desc="Updating clone classes"):
        centroid_id = item['centroid_id']
        function_id = item['function_id']
        clone_type = item['clone_type']
        
        try:
            clone_class = clone_classes[centroid_id]
        except IndexError:
            raise ValueError(f"(error) [import_llm_results] centroid_id {centroid_id} index out of range for clone list")

        try:
            clone_class.suspicious_functions.remove(function_id)
        except ValueError as e:
            raise ValueError(f"(error) [import_llm_results] Failed to remove function_id {function_id} from suspicious_functions of clone class {centroid_id}: {e}")
        
        if clone_type != 0:
            clone_class.confirmed_functions.append(function_id)

    print("(info) [import_llm_results] Clone classes updated successfully.")
    return clone_classes

if __name__ == "__main__":
    # 示例用法
    file_path = config.bd_merged_filepath
    results = process_llm_results_file(file_path)
    clone_classes = update_clone_classes(results)
    
    # 将更新后的克隆类列表持久化保存
    output_filepath = config.icc_cache_filepath
    print(f"(info) [import_llm_results] Saving updated clone classes to {output_filepath}...")
    with open(output_filepath, 'wb') as f:
        pickle.dump(clone_classes, f)
    print("(info) [import_llm_results] Save complete.")

    