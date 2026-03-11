import os
import config
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from extract_functions import extract_functions_from_files

def generate_and_save_embeddings(output_file=None):
    if output_file is None:
        output_file = os.path.expanduser(config.ef_cache_filepath)
        
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"(info) [embedding] Loading model {checkpoint} on {device}...")

    if device == "cpu":
        print(f"(warning) [embedding] No GPU detected. This process may be slow. Consider using a GPU for faster embedding generation.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    model.eval()

    # 使用 PyTorch 2.0 的图编译加速
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)

    print(f"(info) [embedding] Extracting functions from files...")
    functions = extract_functions_from_files()
        
    # 按代码长度降序排序，预分配最大显存以避免碎片化，处理速度会从慢变快
    sorted_functions = sorted(functions, key=lambda f: len(f.code_snippet), reverse=True)
        
    # 动态批大小：设定显存允许的基准最大 Token 面积
    max_tokens_per_batch = config.emb_batch_size * 512
    batches = []
    current_batch = []
    current_max_len = 0

    for func in sorted_functions:
        # 预估 token 长度（假设4个字符约等于1个Token，最长截断边界为512）
        approx_tokens = min(512, len(func.code_snippet) // 4 + 1)
        new_max_len = max(current_max_len, approx_tokens)
        
        # 确保 当前批次总容量 (batch_size * sequence_length) 不超过上限
        if current_batch and new_max_len * (len(current_batch) + 1) > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = [func]
            current_max_len = approx_tokens
        else:
            current_batch.append(func)
            current_max_len = new_max_len
    if current_batch:
        batches.append(current_batch)

    with torch.no_grad():
        with torch.autocast("cuda"): # 开启 PyTorch 自动混合精度加速 (FP16/BF16)
            for batch_funcs in tqdm(batches, desc="Embedding functions"):
                batch_snippets = [func.code_snippet for func in batch_funcs]
                inputs = tokenizer(
                    batch_snippets, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    pad_to_multiple_of=8
                ).to(device)
                
                # 传入 **inputs 以应用 attention_mask，避免对 pad token 产生多余注意力计算和噪声
                embeddings = model(**inputs)[0]
                
                for j, func in enumerate(batch_funcs):
                    func.embedding = embeddings[j].cpu()
    
    print(f"(info) [embedding] Saving functions with embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(functions, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
