"""
Tuthand Token Optimization Tools
Performance optimization for prompt processing and token management
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TokenMetrics:
    """Token usage metrics for optimization tracking"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    response_time: float
    strategy_used: str
    confidence_score: float

class ContextCompressor:
    """Context compression to reduce token usage while preserving meaning"""
    
    def __init__(self, max_context_tokens: int = 2000):
        self.max_context_tokens = max_context_tokens
        self.compression_ratio = 0.3  # Target 30% of original size
        
    def compress_context(self, context: str, preserve_keywords: List[str] = None) -> str:
        """
        Compress context while preserving essential information
        
        Args:
            context: Original context text
            preserve_keywords: Keywords that must be preserved
            
        Returns:
            Compressed context string
        """
        if not preserve_keywords:
            preserve_keywords = []
            
        # Remove redundant whitespace
        context = re.sub(r'\s+', ' ', context.strip())
        
        # Extract key sentences containing preserve_keywords
        sentences = context.split('.')
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword.lower() in sentence.lower() for keyword in preserve_keywords):
                key_sentences.append(sentence)
        
        # If we have key sentences, prioritize them
        if key_sentences:
            compressed = '. '.join(key_sentences[:5])  # Top 5 key sentences
        else:
            # Fallback to first and last parts
            words = context.split()
            if len(words) > 100:
                first_part = ' '.join(words[:50])
                last_part = ' '.join(words[-50:])
                compressed = f"{first_part} ... {last_part}"
            else:
                compressed = context
        
        return compressed
    
    def semantic_summarization(self, text: str, max_tokens: int = 500) -> str:
        """
        Summarize text while preserving semantic meaning
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summarized text
        """
        # Simple extractive summarization
        sentences = text.split('.')
        
        # Score sentences by keyword density and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Simple scoring: length + position weight
            score = len(sentence.split()) + (1.0 / (i + 1))  # Earlier sentences get higher score
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        
        summary_parts = []
        token_count = 0
        
        for score, sentence in scored_sentences:
            estimated_tokens = len(sentence.split()) * 1.3  # Rough token estimation
            if token_count + estimated_tokens <= max_tokens:
                summary_parts.append(sentence)
                token_count += estimated_tokens
            else:
                break
        
        return '. '.join(summary_parts) + '.'

class PromptOptimizer:
    """Optimize prompts for performance and cost efficiency"""
    
    def __init__(self):
        self.token_weights = {
            'system': 1.0,
            'user': 1.2,
            'assistant': 1.5
        }
        
    def optimize_prompt_structure(self, prompt: str, strategy: str) -> str:
        """
        Optimize prompt structure for specific strategy
        
        Args:
            prompt: Original prompt text
            strategy: Strategy being used (plain, chain_of_thought, etc.)
            
        Returns:
            Optimized prompt
        """
        if strategy == 'plain':
            return self._optimize_for_plain(prompt)
        elif strategy == 'chain_of_thought':
            return self._optimize_for_cot(prompt)
        elif strategy == 'react':
            return self._optimize_for_react(prompt)
        elif strategy == 'reflection':
            return self._optimize_for_reflection(prompt)
        else:
            return prompt
    
    def _optimize_for_plain(self, prompt: str) -> str:
        """Optimize for plain response strategy"""
        # Remove unnecessary explanation requests
        prompt = re.sub(r'explain in detail|provide detailed explanation', 'explain', prompt)
        prompt = re.sub(r'step by step|step-by-step', '', prompt)
        
        # Add efficiency instructions
        return f"{prompt}\n\nProvide a direct, concise answer."
    
    def _optimize_for_cot(self, prompt: str) -> str:
        """Optimize for chain-of-thought strategy"""
        # Add structured thinking instructions
        return f"{prompt}\n\nThink through this step by step:\n1. Analysis\n2. Reasoning\n3. Conclusion"
    
    def _optimize_for_react(self, prompt: str) -> str:
        """Optimize for ReAct strategy"""
        # Add ReAct framework
        return f"{prompt}\n\nUse this format:\nThought: [reasoning]\nAction: [what to do]\nObservation: [results]\nAnswer: [final response]"
    
    def _optimize_for_reflection(self, prompt: str) -> str:
        """Optimize for reflection strategy"""
        # Add reflection instructions
        return f"{prompt}\n\nProvide initial answer, then reflect on accuracy and completeness."

class CacheManager:
    """Manage prompt and response caching for performance"""
    
    def __init__(self, max_cache_size: int = 10000):
        self.response_cache = {}
        self.context_cache = {}
        self.template_cache = {}
        self.max_cache_size = max_cache_size
        
    def generate_cache_key(self, query: str, context: str, user_type: str) -> str:
        """Generate cache key for query/context combination"""
        combined = f"{query}|{context}|{user_type}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached response if available"""
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            
            # Check if cache is still valid (24 hours)
            if datetime.now() - cached_item['timestamp'] < timedelta(hours=24):
                return cached_item['response']
        
        return None
    
    def cache_response(self, cache_key: str, response: Dict, confidence: float):
        """Cache response with metadata"""
        if len(self.response_cache) >= self.max_cache_size:
            # Remove oldest items
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k]['timestamp'])
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = {
            'response': response,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'hit_count': 0
        }
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        if not self.response_cache:
            return {'hit_rate': 0, 'size': 0, 'avg_confidence': 0}
        
        total_hits = sum(item['hit_count'] for item in self.response_cache.values())
        total_items = len(self.response_cache)
        avg_confidence = sum(item['confidence'] for item in self.response_cache.values()) / total_items
        
        return {
            'hit_rate': total_hits / (total_hits + total_items) if total_hits > 0 else 0,
            'size': total_items,
            'avg_confidence': avg_confidence
        }

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_targets = {
            'response_time': 2.0,
            'token_usage': 200,
            'cost_per_interaction': 0.05
        }
    
    def record_interaction(self, metrics: TokenMetrics):
        """Record interaction metrics"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only last 1000 interactions
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            item for item in self.metrics_history 
            if item['timestamp'] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_response_time = sum(item['metrics'].response_time for item in recent_metrics) / len(recent_metrics)
        avg_tokens = sum(item['metrics'].total_tokens for item in recent_metrics) / len(recent_metrics)
        avg_cost = sum(item['metrics'].cost_estimate for item in recent_metrics) / len(recent_metrics)
        avg_confidence = sum(item['metrics'].confidence_score for item in recent_metrics) / len(recent_metrics)
        
        # Strategy distribution
        strategy_counts = {}
        for item in recent_metrics:
            strategy = item['metrics'].strategy_used
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'period_hours': hours,
            'total_interactions': len(recent_metrics),
            'avg_response_time': avg_response_time,
            'avg_tokens': avg_tokens,
            'avg_cost': avg_cost,
            'avg_confidence': avg_confidence,
            'strategy_distribution': strategy_counts,
            'performance_alerts': self._check_performance_alerts(avg_response_time, avg_tokens, avg_cost)
        }
    
    def _check_performance_alerts(self, avg_response_time: float, avg_tokens: float, avg_cost: float) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        if avg_response_time > self.performance_targets['response_time']:
            alerts.append(f"High response time: {avg_response_time:.2f}s (target: {self.performance_targets['response_time']}s)")
        
        if avg_tokens > self.performance_targets['token_usage']:
            alerts.append(f"High token usage: {avg_tokens:.0f} (target: {self.performance_targets['token_usage']})")
        
        if avg_cost > self.performance_targets['cost_per_interaction']:
            alerts.append(f"High cost: ${avg_cost:.4f} (target: ${self.performance_targets['cost_per_interaction']})")
        
        return alerts

class TokenBudgetManager:
    """Manage token budgets for different strategies and user types"""
    
    def __init__(self):
        self.strategy_budgets = {
            'plain': 150,
            'chain_of_thought': 300,
            'react': 400,
            'reflection': 400,
            'escalation': 125
        }
        
        self.user_type_modifiers = {
            'founder': 1.2,
            'customer': 1.0,
            'developer': 1.5,
            'support': 1.1
        }
    
    def get_token_budget(self, strategy: str, user_type: str) -> int:
        """Get token budget for strategy and user type combination"""
        base_budget = self.strategy_budgets.get(strategy, 200)
        modifier = self.user_type_modifiers.get(user_type, 1.0)
        return int(base_budget * modifier)
    
    def check_budget_compliance(self, actual_tokens: int, strategy: str, user_type: str) -> Dict:
        """Check if actual usage complies with budget"""
        budget = self.get_token_budget(strategy, user_type)
        compliance = actual_tokens <= budget
        
        return {
            'compliant': compliance,
            'budget': budget,
            'actual': actual_tokens,
            'utilization': actual_tokens / budget,
            'overage': max(0, actual_tokens - budget)
        }

# Example usage and testing
if __name__ == "__main__":
    # Test token optimization
    compressor = ContextCompressor()
    optimizer = PromptOptimizer()
    cache_manager = CacheManager()
    monitor = PerformanceMonitor()
    budget_manager = TokenBudgetManager()
    
    # Example context compression
    sample_context = """
    This is a very long context that contains a lot of information about the product.
    It includes pricing details, feature descriptions, technical specifications, and user testimonials.
    The context is quite verbose and could be compressed for better performance.
    We want to preserve the essential information while reducing token usage.
    """
    
    compressed = compressor.compress_context(sample_context, preserve_keywords=['pricing', 'features'])
    print(f"Original length: {len(sample_context.split())}")
    print(f"Compressed length: {len(compressed.split())}")
    print(f"Compression ratio: {len(compressed.split()) / len(sample_context.split()):.2f}")
    
    # Example prompt optimization
    sample_prompt = "Tell me about the product features and explain how they work in detail"
    optimized = optimizer.optimize_prompt_structure(sample_prompt, 'plain')
    print(f"\nOriginal prompt: {sample_prompt}")
    print(f"Optimized prompt: {optimized}")
    
    # Example budget checking
    budget_check = budget_manager.check_budget_compliance(250, 'chain_of_thought', 'developer')
    print(f"\nBudget compliance: {budget_check}")