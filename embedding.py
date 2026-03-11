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

    print(f"(info) [embedding] Extracting functions from files...")
    functions = extract_functions_from_files()
        
    # 按代码长度排序（Smart Batching），最大限度减少填充（Padding）导致的无用算力消耗
    sorted_functions = sorted(functions, key=lambda f: len(f.code_snippet))
        
    batch_size = config.emb_batch_size
    with torch.no_grad():
        with torch.autocast("cuda"): # 开启 PyTorch 自动混合精度加速 (FP16/BF16)
            for i in tqdm(range(0, len(sorted_functions), batch_size), desc="Embedding functions"):
                batch_funcs = sorted_functions[i:i + batch_size]
                batch_snippets = [func.code_snippet for func in batch_funcs]
                inputs = tokenizer(batch_snippets, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                
                # 传入 **inputs 以应用 attention_mask，避免对 pad token 产生多余注意力计算和噪声
                embeddings = model(**inputs)[0]
                
                for j, func in enumerate(batch_funcs):
                    func.embedding = embeddings[j].cpu()
    
    print(f"(info) [embedding] Saving functions with embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(functions, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
