#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import config


def load_batch_ids(file_path: str) -> List[str]:
	"""
	从文件读取 batch_id 列表（每行一个）。

	Args:
		file_path: batch_id 文件路径

	Returns:
		batch_id 列表（按行读取，去除空行与首尾空白）
	"""
	if not file_path:
		raise ValueError("file_path 不能为空")
	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"文件不存在: {file_path}")

	with open(file_path, "r", encoding="utf-8") as file_handle:
		return [line.strip() for line in file_handle if line.strip()]


def _parse_filename_from_content_disposition(content_disposition: Optional[str]) -> Optional[str]:
	"""从 Content-Disposition 中解析文件名。"""
	if not content_disposition:
		return None
	match = re.search(r"filename\*=([^']*)''([^;]+)", content_disposition, flags=re.IGNORECASE)
	if match:
		return urllib.parse.unquote(match.group(2))
	match = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
	if match:
		return match.group(1)
	return None


def retrieve_batch(
	batch_id: str,
	api_key: Optional[str] = None,
	base_url: str = "https://open.bigmodel.cn/api/paas/v4",
	timeout: int = 30,
) -> Dict[str, Any]:
	"""
	获取批处理任务详情（Batch 对象）。

	Args:
		batch_id: 批处理任务ID
		api_key: 智谱API密钥（可传入或通过环境变量 ZAI_API_KEY / ZHIPU_API_KEY 获取）
		base_url: API 基础地址
		timeout: 请求超时（秒）

	Returns:
		Batch 对象（dict）
	"""
	if not batch_id:
		raise ValueError("batch_id 不能为空")

	resolved_api_key = api_key or os.getenv("ZAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
	if not resolved_api_key:
		raise ValueError("api_key 不能为空，或请设置 ZAI_API_KEY / ZHIPU_API_KEY 环境变量")

	url = f"{base_url}/batches/{batch_id}"
	headers = {
		"Authorization": f"Bearer {resolved_api_key}",
		"Content-Type": "application/json",
	}
	request = urllib.request.Request(url, headers=headers, method="GET")

	try:
		with urllib.request.urlopen(request, timeout=timeout) as response:
			payload = response.read().decode("utf-8")
	except urllib.error.HTTPError as exc:
		error_body = exc.read().decode("utf-8") if exc.fp else ""
		raise RuntimeError(f"请求失败: HTTP {exc.code} {exc.reason}. {error_body}") from exc
	except urllib.error.URLError as exc:
		raise RuntimeError(f"请求失败: {exc.reason}") from exc

	try:
		return json.loads(payload)
	except json.JSONDecodeError as exc:
		raise RuntimeError("响应不是有效的 JSON") from exc


def get_output_file_id(
	batch_id: str,
	api_key: Optional[str] = None,
	base_url: str = "https://open.bigmodel.cn/api/paas/v4",
	timeout: int = 30,
) -> Optional[str]:
	"""
	输入 batch_id，输出 output_file_id。

	Args:
		batch_id: 批处理任务ID
		api_key: 智谱API密钥（可传入或通过环境变量 ZAI_API_KEY / ZHIPU_API_KEY 获取）
		base_url: API 基础地址
		timeout: 请求超时（秒）

	Returns:
		output_file_id（如果不存在则返回 None）
	"""
	batch_obj = retrieve_batch(
		batch_id=batch_id,
		api_key=api_key,
		base_url=base_url,
		timeout=timeout,
	)
	output_file_id = batch_obj.get("output_file_id")
	if output_file_id is None and "data" in batch_obj and isinstance(batch_obj["data"], dict):
		output_file_id = batch_obj["data"].get("output_file_id")
	return output_file_id


def get_output_file_ids(
	batch_ids: List[str],
	api_key: Optional[str] = None,
	base_url: str = "https://open.bigmodel.cn/api/paas/v4",
	timeout: int = 30,
) -> List[Optional[str]]:
	"""
	输入 batch_id 列表，输出 output_file_id 列表。

	Args:
		batch_ids: 批处理任务ID列表
		api_key: 智谱API密钥（可传入或通过环境变量 ZAI_API_KEY / ZHIPU_API_KEY 获取）
		base_url: API 基础地址
		timeout: 请求超时（秒）

	Returns:
		output_file_id 列表（对应位置如不存在则为 None）
	"""
	if not isinstance(batch_ids, list):
		raise ValueError("batch_ids 必须为列表")
	return [
		get_output_file_id(
			batch_id=batch_id,
			api_key=api_key,
			base_url=base_url,
			timeout=timeout,
		)
		for batch_id in batch_ids
	]


def download_output_file(
	file_id: str,
	output_dir: str,
	api_key: Optional[str] = None,
	base_url: str = "https://open.bigmodel.cn/api/paas/v4",
	timeout: int = 30,
	filename: Optional[str] = None,
) -> str:
	"""
	根据 output_file_id 下载文件到指定目录。

	Args:
		file_id: 输出文件ID（output_file_id）
		output_dir: 保存目录
		api_key: 智谱API密钥（可传入或通过环境变量 ZAI_API_KEY / ZHIPU_API_KEY 获取）
		base_url: API 基础地址
		timeout: 请求超时（秒）
		filename: 自定义文件名（可选；不传则尝试从响应头解析）

	Returns:
		保存的文件路径
	"""
	if not file_id:
		raise ValueError("file_id 不能为空")
	if not output_dir:
		raise ValueError("output_dir 不能为空")

	resolved_api_key = api_key or os.getenv("ZAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
	if not resolved_api_key:
		raise ValueError("api_key 不能为空，或请设置 ZAI_API_KEY / ZHIPU_API_KEY 环境变量")

	os.makedirs(output_dir, exist_ok=True)

	url = f"{base_url}/files/{file_id}/content"
	headers = {
		"Authorization": f"Bearer {resolved_api_key}",
	}
	request = urllib.request.Request(url, headers=headers, method="GET")

	try:
		with urllib.request.urlopen(request, timeout=timeout) as response:
			if not filename:
				content_disposition = response.headers.get("Content-Disposition")
				filename = _parse_filename_from_content_disposition(content_disposition) or file_id
			if not filename:
				filename = file_id
			base_name, ext = os.path.splitext(filename)
			if not ext:
				ext = ".jsonl"
			candidate = f"{base_name}{ext}"
			output_path = os.path.join(output_dir, candidate)
			if os.path.exists(output_path):
				candidate = f"{base_name}_{file_id}{ext}"
				output_path = os.path.join(output_dir, candidate)
				if os.path.exists(output_path):
					counter = 1
					while True:
						candidate = f"{base_name}_{file_id}_{counter}{ext}"
						output_path = os.path.join(output_dir, candidate)
						if not os.path.exists(output_path):
							break
						counter += 1
			with open(output_path, "wb") as file_handle:
				while True:
					chunk = response.read(8192)
					if not chunk:
						break
					file_handle.write(chunk)
			return output_path
	except urllib.error.HTTPError as exc:
		error_body = exc.read().decode("utf-8") if exc.fp else ""
		raise RuntimeError(f"请求失败: HTTP {exc.code} {exc.reason}. {error_body}") from exc
	except urllib.error.URLError as exc:
		raise RuntimeError(f"请求失败: {exc.reason}") from exc


def download_output_files(
	file_ids: List[str],
	output_dir: str,
	api_key: Optional[str] = None,
	base_url: str = "https://open.bigmodel.cn/api/paas/v4",
	timeout: int = 30,
) -> List[Optional[str]]:
	"""
	批量下载 output_file_id 对应的文件。

	Args:
		file_ids: output_file_id 列表
		output_dir: 保存目录
		api_key: 智谱API密钥（可传入或通过环境变量 ZAI_API_KEY / ZHIPU_API_KEY 获取）
		base_url: API 基础地址
		timeout: 请求超时（秒）

	Returns:
		下载后的文件路径列表（对应位置如 file_id 为空则为 None）
	"""
	if not isinstance(file_ids, list):
		raise ValueError("file_ids 必须为列表")
	if not output_dir:
		raise ValueError("output_dir 不能为空")

	results: List[Optional[str]] = []
	for file_id in file_ids:
		if not file_id:
			results.append(None)
			continue
		results.append(
			download_output_file(
				file_id=file_id,
				output_dir=output_dir,
				api_key=api_key,
				base_url=base_url,
				timeout=timeout,
			)
		)
	return results


def merge_files_and_delete_sources(
	input_files: List[str],
	output_file: str,
	chunk_size: int = 1024 * 1024,
) -> str:
	"""
	将多个文件按顺序合并为一个文件，并删除原文件。

	Args:
		input_files: 待合并的文件路径列表
		output_file: 合并后的输出文件路径
		chunk_size: 读写块大小（字节）

	Returns:
		输出文件路径
	"""
	if not isinstance(input_files, list) or not input_files:
		raise ValueError("input_files 必须为非空列表")
	if not output_file:
		raise ValueError("output_file 不能为空")
	if chunk_size <= 0:
		raise ValueError("chunk_size 必须为正数")

	output_dir = os.path.dirname(output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(output_file, "wb") as out_handle:
		for file_path in input_files:
			if not file_path:
				raise ValueError("input_files 中存在空路径")
			if not os.path.isfile(file_path):
				raise FileNotFoundError(f"文件不存在: {file_path}")
			with open(file_path, "rb") as in_handle:
				while True:
					chunk = in_handle.read(chunk_size)
					if not chunk:
						break
					out_handle.write(chunk)

	for file_path in input_files:
		os.remove(file_path)

	return output_file


if __name__ == "__main__":
    print("开始下载文件。")
    print("正在加载 batch_id 列表...")
    batch_ids = load_batch_ids(config.bu_batch_ids_filepath)
    print(f"共加载 {len(batch_ids)} 个 batch_id。")
    print("正在获取 output_file_id 列表...")
    output_file_ids = get_output_file_ids(batch_ids)
    print("正在下载输出文件...")
    downloaded_files = download_output_files(output_file_ids, config.bd_download_dir)
    successful_downloads = [f for f in downloaded_files if f]
    print(f"共下载 {len(successful_downloads)} 个文件，保存在目录: {config.bd_download_dir}")
    if not successful_downloads:
        print("没有成功下载的文件，退出。")
        exit(0)
    print(f"正在合并文件到: {config.bd_merged_filepath} ...")
    merge_files_and_delete_sources(successful_downloads, config.bd_merged_filepath)
    print("文件合并完成。")
    print(f"合并后的文件保存在: {config.bd_merged_filepath}")
    if os.path.isdir(config.bd_download_dir):
        shutil.rmtree(config.bd_download_dir)
        print(f"已删除目录: {config.bd_download_dir}")
		