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

    print(f"(info) [embedding] Extracting functions from files...")
    functions = extract_functions_from_files()
        
    with torch.no_grad():
        for func in tqdm(functions, desc="Embedding functions"):
            inputs = tokenizer.encode(func.code_snippet, return_tensors="pt").to(device)
            embedding = model(inputs)[0]
            # 这里将 embedding 放到 CPU 并保存以避免 GPU 内存打满，并确保能正常被 pickle 序列化
            func.embedding = embedding.cpu()
    
    print(f"(info) [embedding] Saving functions with embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(functions, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
