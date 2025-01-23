import sys
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Removed BitsAndBytesConfig import since we're not using quantization

# Centralized model configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Change your LLM Model here
DEVICE = "auto"

class LLMHandler:
    def __init__(self):
        self.pipe = None

    def initialize_llm(self) -> None:
        """Initialize the LLM model."""
        try:
            print(f"Loading model {MODEL_ID}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map=DEVICE
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                padding_side="left"
            )
            
            # Set pad token to eos token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            print("Model initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            sys.exit(1)

    def get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from LLM with error handling."""
        try:
            if self.pipe is None:
                self.initialize_llm()

            # Updated pipeline parameters to match the correct API
            response = self.pipe(
                prompt,
                max_new_tokens=100,  # Changed from max_length to max_new_tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                return_full_text=True
            )
            
            response_text = response[0]['generated_text']
            
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            if 'Output:' in response_text:
                response_text = response_text.split('Output:')[-1].strip()
            
            valid_types = ['QuoteRequest', 'BondRequest', 'GENERAL']
            for type_ in valid_types:
                if type_ in response_text:
                    return type_
            
            return "GENERAL"
            
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return "GENERAL"
