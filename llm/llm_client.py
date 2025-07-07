# llm/llm_client.py

from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass
from llm.prompt_templates import prompt_templates
@dataclass
class LLMResponse:
    text: str
    confidence: float

class LLMClient:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize Flan-T5 model and tokenizer"""
        print(f"Loading {model_name}...")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_length = 512
        self.min_length = 50
        
        print("Model loaded successfully!")

        # Define prompt templates
       
    def generate_response(self, query: str, context: List[str], query_type: str = 'information') -> LLMResponse:
        """
        Generate response using Flan-T5
        
        Args:
            query: User query
            context: Retrieved context passages
            query_type: Type of query (data_access, information, technical)
        """
        try:
            # Select appropriate template
            template = prompt_templates.get(query_type, prompt_templates['information'])
            
            # Prepare prompt
            context_text = "\n".join(context)
            prompt = template.format(
                context=context_text,
                query=query
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=True
            )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence based on output probabilities
            confidence = self._calculate_confidence(outputs)
            
            return LLMResponse(
                text=response_text,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return LLMResponse(
                text="I apologize, but I encountered an error generating the response. Please try rephrasing your question.",
                confidence=0.0
            )

    def _calculate_confidence(self, outputs) -> float:
        """Calculate confidence score based on output probabilities"""
        with torch.no_grad():
            # Get output probabilities
            logits = self.model.get_output_embeddings()(outputs)
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate mean probability of selected tokens
            mean_prob = torch.mean(torch.max(probs, dim=-1)[0])
            
            return float(mean_prob)

    def _preprocess_context(self, context: List[str]) -> str:
        """Preprocess and format context for the model"""
        # Limit context length
        total_length = 0
        selected_contexts = []
        
        for ctx in context:
            # Approximate token count
            token_count = len(ctx.split())
            if total_length + token_count > self.max_length * 0.7:  # Leave room for query and response
                break
            selected_contexts.append(ctx)
            total_length += token_count
        
        return "\n".join(selected_contexts)

def main():
    """Test the LLM client"""
    # Initialize client
    llm = LLMClient()
    
    # Test queries
    test_queries = [
        {
            "query": "How can I download SST data?",
            "context": [
                "SST data can be downloaded from MOSDAC portal after registration.",
                "Users need to log in and navigate to the SST products section.",
                "Data is available in NetCDF format with daily and monthly averages."
            ],
            "type": "data_access"
        },
        {
            "query": "What is the resolution of INSAT imagery?",
            "context": [
                "INSAT-3D imager provides resolution of 1km for visible band.",
                "Thermal infrared bands have resolution of 4km.",
                "Data is available in multiple spectral bands."
            ],
            "type": "technical"
        }
    ]
    
    # Generate responses
    for test in test_queries:
        print(f"\nQuery: {test['query']}")
        response = llm.generate_response(
            query=test['query'],
            context=test['context'],
            query_type=test['type']
        )
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence:.2f}")

if __name__ == "__main__":
    main()