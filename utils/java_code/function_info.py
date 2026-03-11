from dataclasses import dataclass


@dataclass
class FunctionInfo:
    start_line: int
    end_line: int
    code_snippet: str
    path: str
