from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal


class ExperimentInput(BaseModel):
    initial_sentence_pool: List[str] = Field(..., min_length=100)
    num_layers: int = Field(..., ge=2, le=6)
    num_nodes: int = Field(..., ge=3, le=10)
    num_output_sentences: int = Field(..., ge=3, le=7)
    prompt_template_id: Literal['creative', 'neutral', 'chain_of_thought']
    resume_from_checkpoint: bool = False
    config_overrides: Optional[Dict[str, Any]] = None
    
    @field_validator('num_output_sentences')
    @classmethod
    def validate_output_sentences(cls, v, info):
        num_nodes = info.data.get('num_nodes', 0)
        initial_sentence_pool = info.data.get('initial_sentence_pool', [])
        if num_nodes * v > len(initial_sentence_pool):
            raise ValueError(
                f"num_nodes * num_output_sentences ({num_nodes * v}) cannot exceed "
                f"initial_sentence_pool length ({len(initial_sentence_pool)})"
            )
        # FIXED: Validate that num_nodes == num_output_sentences for semantic mixing
        if num_nodes != v:
            raise ValueError(
                f"num_nodes ({num_nodes}) must equal num_output_sentences ({v}) for semantic mixing to work correctly."
            )
        return v
