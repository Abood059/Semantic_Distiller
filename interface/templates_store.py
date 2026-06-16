PROMPT_TEMPLATES = {
    "creative": "Generate creative variations of the following sentences: {sentences}",
    "neutral": "Generate neutral variations of the following sentences: {sentences}",
    "chain_of_thought": "Think step by step and generate variations of the following sentences: {sentences}"
}


def get_template(template_id: str) -> str:
    """Get a prompt template by ID."""
    if template_id not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template_id: {template_id}. Available: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[template_id]
