import os
import config
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from extract_functions import extract_functions_from_files
from utils.data_class.function_embedding import FunctionEmbedding

def generate_and_save_embeddings():
    output_file = os.path.expanduser(config.emb_cache_filepath)
    
    print(f"(info) [embedding] Extracting functions from files...")
    functions = extract_functions_from_files()

    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"(info) [embedding] Loading model {checkpoint} on {device}...")

    if device == "cpu":
        print(f"(warning) [embedding] No GPU detected. This process may be slow. Consider using a GPU for faster embedding generation.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    model.eval()

    fe = []

    with torch.inference_mode():  
        for func in tqdm(functions, desc="Embedding functions"):  
            inputs = tokenizer.encode(
                func.code_snippet, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)  
            
            embedding = model(inputs)[0]
            
            fe.append(FunctionEmbedding(id=func.id, embedding=embedding.squeeze().cpu().numpy()))

    fe.sort(key=lambda x: x.id)

    print(f"(info) [embedding] Saving functions with embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(fe, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
