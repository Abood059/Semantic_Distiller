from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class NodeOutput(BaseModel):
    node_id: int
    generated_texts: List[str]
    embeddings: List[List[float]]
    intra_node_diversity: float


class LayerOutput(BaseModel):
    layer_id: int
    nodes: List[NodeOutput]
    layer_coverage_score: float
    layer_bias_vector: Optional[List[float]] = None


class ExperimentOutput(BaseModel):
    experiment_id: Optional[str] = None
    layers: List[LayerOutput]
    metadata: Dict[str, Any]
