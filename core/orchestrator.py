import logging
import gc
import torch
import numpy as np
from typing import Dict, Any
from interface.input_schemas import ExperimentInput
from interface.output_schemas import ExperimentOutput, LayerOutput, NodeOutput
from interface.templates_store import get_template
from core.storage.db_handler import DatabaseHandler
from core.storage.checkpoint_manager import is_node_done
from core.services.embedding_service import EmbeddingService
from core.services.generation_service import GenerationService
from core.services.sampling_service import FPSSampler


class SADNOrchestrator:
    def __init__(
        self,
        config: Dict[str, Any],
        db_handler: DatabaseHandler,
        embedding_service: EmbeddingService,
        generation_service: GenerationService,
        sampler: FPSSampler
    ):
        """Initialize the SADN orchestrator.
        
        Args:
            config: System configuration dictionary
            db_handler: Database handler for persistence
            embedding_service: Service for generating embeddings
            generation_service: Service for text generation
            sampler: Service for diverse sampling
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.db_handler = db_handler
        self.embedding_service = embedding_service
        self.generation_service = generation_service
        self.sampler = sampler
    
    def run(self, request: ExperimentInput) -> ExperimentOutput:
        """Run the SADN experiment.
        
        Args:
            request: Experiment input parameters
            
        Returns:
            ExperimentOutput with all layer/node results
        """
        self.logger.info(f"Starting experiment with {request.num_layers} layers, {request.num_nodes} nodes")
        
        # Phase 1: Initial Input Preparation
        self.logger.info("Phase 1: Initial input preparation")
        initial_embeddings = self.embedding_service.embed(request.initial_sentence_pool)
        
        # Select diverse sentences for layer 1
        total_needed = request.num_nodes * request.num_output_sentences
        selected_indices = self.sampler.select(initial_embeddings, total_needed)
        
        # Distribute selected sentences to layer 1 nodes
        layer1_inputs = []
        for node in range(request.num_nodes):
            start_idx = node * request.num_output_sentences
            end_idx = start_idx + request.num_output_sentences
            node_sentences = [request.initial_sentence_pool[i] for i in selected_indices[start_idx:end_idx]]
            layer1_inputs.append(node_sentences)
        
        # Phase 2: Layer/Node Loop
        self.logger.info("Phase 2: Layer/Node processing")
        for layer in range(1, request.num_layers + 1):
            self.logger.info(f"Processing layer {layer}/{request.num_layers}")
            
            for node in range(1, request.num_nodes + 1):
                self.logger.info(f"Processing layer {layer}, node {node}")
                
                # Checkpoint: Skip if already done
                if request.resume_from_checkpoint and is_node_done(self.db_handler, layer, node):
                    self.logger.info(f"Skipping layer {layer}, node {node} (already done)")
                    continue
                
                try:
                    # Fetch inputs
                    if layer == 1:
                        input_sentences = layer1_inputs[node - 1]
                    else:
                        input_sentences = self.db_handler.get_inputs_for_node(
                            layer, node, request.num_nodes
                        )
                    
                    # Build prompt
                    template = get_template(request.prompt_template_id)
                    numbered_sentences = "\n".join(
                        [f"{i+1}. {s}" for i, s in enumerate(input_sentences)]
                    )
                    prompt = template.format(sentences=numbered_sentences)
                    
                    # Generate
                    generated_texts = self.generation_service.generate(
                        prompt=prompt,
                        num_sentences=request.num_output_sentences
                    )
                    
                    # Embed generated texts
                    generated_embeddings = self.embedding_service.embed(generated_texts)
                    
                    # Calculate intra-node diversity (average cosine distance)
                    intra_diversity = self._calculate_intra_diversity(generated_embeddings)
                    
                    # Save results
                    self.db_handler.save_node_results(
                        layer=layer,
                        node=node,
                        sentences=generated_texts,
                        embeddings=generated_embeddings
                    )
                    
                    self.logger.info(
                        f"Completed layer {layer}, node {node} "
                        f"(intra-diversity: {intra_diversity:.4f})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error in layer {layer}, node {node}: {e}")
                    # Save error to database and continue
                    error_text = f"ERROR: {str(e)}"
                    self.db_handler.save_node_results(
                        layer=layer,
                        node=node,
                        sentences=[error_text] * request.num_output_sentences,
                        embeddings=[[0.0] * 384] * request.num_output_sentences
                    )
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Phase 3: Final Assembly
        self.logger.info("Phase 3: Final assembly")
        results = self._assemble_results(request)
        
        return results
    
    def _calculate_intra_diversity(self, embeddings: list[list[float]]) -> float:
        """Calculate average cosine distance between all pairs of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Average cosine distance
        """
        if len(embeddings) < 2:
            return 0.0
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        # Compute pairwise cosine similarities
        similarities = embeddings_array @ embeddings_array.T
        
        # Convert to distances and average (excluding diagonal)
        n = len(embeddings)
        distances = 1.0 - similarities
        mask = ~np.eye(n, dtype=bool)
        avg_distance = np.mean(distances[mask])
        
        return float(avg_distance)
    
    def _assemble_results(self, request: ExperimentInput) -> ExperimentOutput:
        """Assemble final results from database.
        
        Args:
            request: Original experiment input
            
        Returns:
            ExperimentOutput with all results
        """
        layers = []
        
        for layer in range(1, request.num_layers + 1):
            nodes = []
            layer_embeddings = []
            
            for node in range(1, request.num_nodes + 1):
                # Fetch node results from database
                conn = self.db_handler._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sentence, embedding FROM results
                    WHERE layer = ? AND node = ? AND status = 'DONE'
                    ORDER BY sentence_index
                """, (layer, node))
                rows = cursor.fetchall()
                conn.close()
                
                if rows:
                    sentences = [row[0] for row in rows]
                    embeddings = [
                        self.db_handler._deserialize_embedding(row[1]) for row in rows
                    ]
                    layer_embeddings.extend(embeddings)
                    
                    # Calculate intra-node diversity
                    intra_diversity = self._calculate_intra_diversity(embeddings)
                    
                    node_output = NodeOutput(
                        node_id=node,
                        generated_texts=sentences,
                        embeddings=embeddings,
                        intra_node_diversity=intra_diversity
                    )
                    nodes.append(node_output)
            
            # Calculate layer metrics
            if layer_embeddings:
                layer_coverage = self._calculate_layer_coverage(layer_embeddings)
                layer_bias = self._calculate_layer_bias(layer_embeddings)
            else:
                layer_coverage = 0.0
                layer_bias = None
            
            layer_output = LayerOutput(
                layer_id=layer,
                nodes=nodes,
                layer_coverage_score=layer_coverage,
                layer_bias_vector=layer_bias
            )
            layers.append(layer_output)
        
        return ExperimentOutput(
            experiment_id=None,
            layers=layers,
            metadata={
                "num_layers": request.num_layers,
                "num_nodes": request.num_nodes,
                "num_output_sentences": request.num_output_sentences,
                "prompt_template_id": request.prompt_template_id
            }
        )
    
    def _calculate_layer_coverage(self, embeddings: list[list[float]]) -> float:
        """Calculate average distance between nodes in a layer.
        
        Args:
            embeddings: All embeddings from the layer
            
        Returns:
            Average distance between node centroids
        """
        if not embeddings:
            return 0.0
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        # Compute pairwise distances
        similarities = embeddings_array @ embeddings_array.T
        distances = 1.0 - similarities
        
        # Average all distances
        n = len(embeddings)
        mask = ~np.eye(n, dtype=bool)
        avg_distance = np.mean(distances[mask])
        
        return float(avg_distance)
    
    def _calculate_layer_bias(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate centroid (bias vector) of layer embeddings.
        
        Args:
            embeddings: All embeddings from the layer
            
        Returns:
            Centroid vector as list of floats
        """
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        centroid = np.mean(embeddings_array, axis=0)
        
        return centroid.tolist()
