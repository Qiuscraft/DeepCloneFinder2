from dataclasses import dataclass

import numpy as np


@dataclass
class FunctionEmbedding:
    id: int
    embedding: np.ndarray
