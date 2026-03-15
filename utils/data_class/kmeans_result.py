from dataclasses import dataclass


@dataclass
class KMeansResult:
    id: int
    centroid_id: int
    distance: float
