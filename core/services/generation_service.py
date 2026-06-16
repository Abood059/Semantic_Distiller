import logging
import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class GenerationService:
    def __init__(self, model_name: str, use_4bit: bool = True, **kwargs):
        """Initialize generation service with quantized model.
        
        Args:
            model_name: HuggingFace model identifier
            use_4bit: Whether to use 4-bit quantization
            **kwargs: Additional arguments for model loading
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        self.logger.info(f"Loading generation model: {model_name}")
        
        # Configure 4-bit quantization if requested
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info("Generation model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        num_sentences: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> list[str]:
        """Generate sentences based on prompt.
        
        Args:
            prompt: Input prompt for generation
            num_sentences: Number of sentences to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            List of generated sentences (exactly num_sentences length)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=150,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to extract sentences
        sentences = _post_process_output(generated_text, num_sentences)
        
        self.logger.info(f"Generated {len(sentences)} sentences")
        return sentences
    
    def cleanup(self) -> None:
        """Clean up model and tokenizer to free memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Generation service cleaned up")


def _post_process_output(raw_text: str, num_sentences: int) -> list[str]:
    """Post-process raw generation output to extract sentences.
    
    Args:
        raw_text: Raw text from model generation
        num_sentences: Expected number of sentences
        
    Returns:
        List of sentences (padded to num_sentences if needed)
    """
    # Clean text from common markers
    cleaned = raw_text
    for marker in ["[INST]", "[/INST]", "Human:", "Assistant:", "###"]:
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip()
    
    sentences = []
    
    # Try to extract JSON list first
    json_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    if json_match:
        try:
            sentences = json.loads(json_match.group(0))
            if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
                if len(sentences) >= num_sentences:
                    return sentences[:num_sentences]
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: extract numbered list
    numbered_pattern = r'(?:^|\n)\s*(\d+\.|-)\s*([^\n]+)'
    matches = re.findall(numbered_pattern, cleaned, re.MULTILINE)
    if matches:
        sentences = [match[1].strip() for match in matches]
    
    # If still no sentences, split by newlines and filter
    if not sentences:
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        sentences = lines
    
    # Pad or truncate to exact num_sentences
    while len(sentences) < num_sentences:
        sentences.append("")
    if len(sentences) > num_sentences:
        sentences = sentences[:num_sentences]
    
    return sentences
