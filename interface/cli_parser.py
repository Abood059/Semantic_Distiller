import json
import sys
from pydantic import ValidationError
from .input_schemas import ExperimentInput
from .output_schemas import ExperimentOutput


def read_experiment_request(json_path: str) -> ExperimentInput:
    """Load and validate experiment request from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        return ExperimentInput(**data)
    except ValidationError as e:
        print(f"Validation error in {json_path}:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  Field '{field}': {error['msg']}")
        sys.exit(1)


def write_experiment_results(output_path: str, results: ExperimentOutput) -> None:
    """Write experiment results to JSON file."""
    with open(output_path, 'w') as f:
        f.write(results.model_dump_json(indent=2))
