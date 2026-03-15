#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from zai import ZhipuAiClient

import config


processing_filepath = config.gbr_output_file
batch_ids_output_filepath = config.bu_batch_ids_filepath

request_parts_dir = config.bu_request_parts_dir
max_part_file_size = config.bu_max_part_file_size
max_part_file_lines = config.bu_max_part_file_lines
zai_api_key = config.zai_api_key


def split_file_by_size_and_lines(
    file_path: str,
    max_file_size: int,
    max_lines: int,
    output_dir: str,
    encoding: str = "utf-8",
) -> List[str]:
    """
    将单个文件按最大大小与最大行数拆分为多个文件。

    Args:
        file_path: 输入文件路径
        max_file_size: 单文件最大大小（字节）
        max_lines: 单文件最大行数
        output_dir: 输出目录
        encoding: 文件编码，默认 utf-8

    Returns:
        拆分后文件路径列表
    """
    if max_file_size <= 0:
        raise ValueError("max_file_size 必须为正整数")
    if max_lines <= 0:
        raise ValueError("max_lines 必须为正整数")

    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    suffix = input_path.suffix

    output_files: List[str] = []
    current_size = 0
    current_lines = 0
    file_index = 1
    outfile = None

    try:
        with open(input_path, "r", encoding=encoding) as infile:
            for line in infile:
                line_bytes = len(line.encode(encoding))

                need_new_file = (
                    outfile is None
                    or (current_size + line_bytes) > max_file_size
                    or current_lines >= max_lines
                )

                if need_new_file:
                    if outfile:
                        outfile.close()

                    output_path = os.path.join(
                        output_dir, f"{base_name}_part{file_index}{suffix}"
                    )
                    outfile = open(output_path, "w", encoding=encoding)
                    output_files.append(output_path)
                    file_index += 1
                    current_size = 0
                    current_lines = 0

                outfile.write(line)
                current_size += line_bytes
                current_lines += 1
    finally:
        if outfile:
            outfile.close()

    return output_files


def upload_files_and_create_batches(
    file_paths: List[str],
    api_key: str,
    endpoint: str = "/v4/chat/completions",
    auto_delete_input_file: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    上传多个JSONL文件并创建批处理任务，返回批处理ID列表。

    Args:
        file_paths: 文件路径列表
        api_key: 智谱API密钥
        endpoint: 批处理端点，默认为 /v4/chat/completions
        auto_delete_input_file: 是否自动删除batch原始文件，默认为True
        metadata: 批处理元数据（最多16个键值对）

    Returns:
        批处理ID列表（按输入文件顺序，失败的文件将被跳过）
    """
    if not file_paths:
        return []

    client = ZhipuAiClient(api_key=api_key)
    batch_ids: List[str] = []

    for file_path in file_paths:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"输入文件不存在: {file_path}")
        if file_path_obj.suffix.lower() != ".jsonl":
            raise ValueError(f"输入文件必须为 .jsonl 格式: {file_path}")

        with open(file_path_obj, "rb") as f:
            file_object = client.files.create(file=f, purpose="batch")
        if not hasattr(file_object, "id"):
            continue
        file_id = file_object.id

        batch = client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,  # type: ignore
            auto_delete_input_file=auto_delete_input_file,
            metadata=metadata or {},
        )
        if hasattr(batch, "id"):
            batch_ids.append(batch.id)

    return batch_ids


def save_batch_ids_to_file(
    batch_ids: List[str],
    output_file: str,
    encoding: str = "utf-8",
) -> None:
    """
    将批处理ID列表保存到指定文件中（每行一个ID）。

    Args:
        batch_ids: 批处理ID列表
        output_file: 输出文件路径
        encoding: 文件编码，默认 utf-8
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=encoding) as f:
        for batch_id in batch_ids:
            f.write(f"{batch_id}\n")


def delete_files(file_paths: List[str]) -> None:
    """
    删除指定文件列表中存在的文件。

    Args:
        file_paths: 待删除文件路径列表
    """
    for file_path in file_paths:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            # Ignore deletion failures to avoid breaking the main flow.
            continue


if __name__ == "__main__":
    batch_file_parts = split_file_by_size_and_lines(processing_filepath, max_part_file_size, max_part_file_lines, request_parts_dir)
    batch_ids = upload_files_and_create_batches(batch_file_parts, zai_api_key)
    save_batch_ids_to_file(batch_ids, batch_ids_output_filepath)
    delete_files(batch_file_parts)
