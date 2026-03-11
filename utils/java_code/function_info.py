from dataclasses import dataclass
import torch

@dataclass
class FunctionInfo:
    start_line: int
    end_line: int
    code_snippet: str
    path: str
    embedding: torch.Tensor 
