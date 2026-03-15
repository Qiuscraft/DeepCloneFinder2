from dataclasses import dataclass
import numpy as np

@dataclass
class FunctionInfo:
    id: int
    start_line: int
    end_line: int
    code_snippet: str
    path: str
