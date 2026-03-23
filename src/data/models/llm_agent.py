from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Dict, List, Tuple
import json

class LightweightLLMAgent:
    """CPU-optimized LLM for risk analysis"""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        print(f"Loading LLM: {model_name}...")
        
        # Load quantized model for CPU efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Sentence transformer for embeddings (runs on CPU)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Risk keywords for zero-shot classification
        self.risk_keywords = {
            'geopolitical': ['strike', 'tariff', 'sanction', 'trade war', 'embargo'],
            'natural_disaster': ['storm', 'flood', 'earthquake', 'hurricane', 'fire'],
            'financial': ['bankruptcy', 'insolvency', 'debt', 'default'],
            'operational': ['quality issue', 'delay', 'shortage', 'recall']
        }
        
        self.generation_pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            device='cpu'
        )
        
    def analyze_news(self, news_text: str) -> Dict:
        """Analyze news article for supply chain risks"""
        
        # Extract entities and risk signals
        prompt = f"""Analyze this supply chain news for risks:
        News: {news_text}
        
        Extract:
        1. Affected entities (suppliers, locations, products)
        2. Risk type (geopolitical/natural/financial/operational)
        3. Severity (1-10)
        4. Recommended action
        
        Output as JSON:"""
        
        try:
            response = self.generation_pipeline(prompt)[0]['generated_text']
            # Extract JSON from response (simplified)
            risk_analysis = self._parse_llm_response(response)
        except:
            risk_analysis = self._rule_based_analysis(news_text)
            
        return risk_analysis
    
    def _rule_based_analysis(self, news_text: str) -> Dict:
        """Fallback rule-based analysis for efficiency"""
        news_lower = news_text.lower()
        
        # Detect risk type
        risk_type = 'unknown'
        for rtype, keywords in self.risk_keywords.items():
            if any(kw in news_lower for kw in keywords):
                risk_type = rtype
                break
                
        # Simple severity scoring
        severity = 5  # default
        if any(word in news_lower for word in ['severe', 'major', 'catastrophic']):
            severity = 8
        elif any(word in news_lower for word in ['minor', 'small', 'limited']):
            severity = 3
            
        return {
            'risk_type': risk_type,
            'severity': severity,
            'entities': self._extract_entities(news_text),
            'confidence': 0.7
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction using embeddings"""
        # Use sentence transformer for similarity
        sentences = text.split('.')
        entity_candidates = []
        
        # Simple heuristic: proper nouns and location names
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entity_candidates.append(word)
                
        return list(set(entity_candidates[:5]))  # Top 5 unique entities
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        try:
            # Try to find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
            
        return {'error': 'Parsing failed'}
    
    def generate_risk_narrative(self, risk_scores: Dict, affected_nodes: List) -> str:
        """Generate human-readable risk narrative"""
        
        narrative_prompt = f"""Generate a concise risk narrative for supply chain:
        High risk nodes: {affected_nodes[:5]}
        Average risk score: {np.mean(list(risk_scores.values())):.2f}
        
        Provide: 2-sentence summary and recommended action."""
        
        try:
            response = self.generation_pipeline(narrative_prompt)[0]['generated_text']
            return response
        except:
            return f"⚠️ Alert: {len(affected_nodes)} nodes at risk. Average risk score: {np.mean(list(risk_scores.values())):.2f}. Consider rerouting shipments."