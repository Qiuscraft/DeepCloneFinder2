from dataclasses import dataclass
import numpy as np

@dataclass
class FunctionInfo:
    start_line: int
    end_line: int
    code_snippet: str
    path: str
    embedding: np.ndarray
