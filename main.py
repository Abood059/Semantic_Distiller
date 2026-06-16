import yaml
import logging
import sys
import argparse
import gc
from interface.cli_parser import read_experiment_request, write_experiment_results
from core.storage.db_handler import DatabaseHandler
from core.services.embedding_service import EmbeddingService
from core.services.sampling_service import FPSSampler
from core.services.generation_service import GenerationService
from core.orchestrator import SADNOrchestrator


def load_system_config(config_path: str = "config.yaml") -> dict:
    """Load system configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Distiller")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    
    try:
        config = load_system_config(args.config)
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("Semantic Distiller initialized successfully")
        logger.info(f"Input file: {args.input}, Output file: {args.output}")
        
        # Read input
        request = read_experiment_request(args.input)
        logger.info(f"Loaded experiment request: {request.num_layers} layers, {request.num_nodes} nodes")
        
        # Initialize database
        db_handler = DatabaseHandler(config['system']['db_path'])
        db_handler.initialize_db()
        logger.info("Database initialized")
        
        # Initialize services
        embedding_service = EmbeddingService(config['system']['embedding_model'])
        generation_service = GenerationService(
            config['system']['generation_model'],
            config['system']['use_4bit']
        )
        sampler = FPSSampler(metric="cosine")
        logger.info("Services initialized")
        
        # Initialize orchestrator
        orchestrator = SADNOrchestrator(
            config, db_handler, embedding_service, generation_service, sampler
        )
        logger.info("Orchestrator initialized")
        
        # Run experiment
        results = orchestrator.run(request)
        logger.info("Experiment completed successfully")
        
        # Write output
        write_experiment_results(args.output, results)
        logger.info(f"Results written to {args.output}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)
