#!/usr/bin/env python3
"""
Tuthand - Week 2 Prototype with Interface Intelligence
Building in Public: Following AI Engineering principles from Chip Huyen's book
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import yaml
import sqlite3
from datetime import datetime
from openai import OpenAI
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    """Trust levels for response routing"""
    AUTO_RUN = "auto_run"
    CONFIRM = "confirm" 
    ESCALATE = "escalate"

@dataclass
class ResponseMetrics:
    """Performance metrics for responses"""
    tokens_used: int
    response_time: float
    cost_estimate: float
    confidence_score: float
    trust_level: TrustLevel

@dataclass
class UserQuery:
    """User query with metadata"""
    text: str
    user_type: str = "customer"
    context: Optional[str] = None
    timestamp: datetime = None

@dataclass
class AssistantResponse:
    """Assistant response with metadata"""
    text: str
    trust_level: TrustLevel
    metrics: ResponseMetrics
    reasoning: Optional[str] = None

class PromptStrategy(Enum):
    """Available prompt strategies"""
    PLAIN = "plain"
    CHAIN_OF_THOUGHT = "cot"
    REACT = "react"
    REFLECT = "reflect"
    REVISE = "revise"

class TuthandAssistant:
    """Main Tuthand AI Assistant with Week 2 features"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.db_path = "performance_metrics.db"
        self.response_cache = {}  # Simple in-memory cache for high-confidence responses
        self.init_database()
        self.load_configurations()
        
    def init_database(self):
        """Initialize SQLite database for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_type TEXT,
                query TEXT,
                response TEXT,
                trust_level TEXT,
                tokens_used INTEGER,
                response_time REAL,
                cost_estimate REAL,
                confidence_score REAL,
                strategy TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def load_configurations(self):
        """Load prompt configurations and strategies"""
        try:
            with open('prompt_config.yaml', 'r') as f:
                self.prompt_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("prompt_config.yaml not found, using defaults")
            self.prompt_config = self.get_default_config()
            
        try:
            with open('prompts/strategies.yaml', 'r') as f:
                self.strategies = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("prompts/strategies.yaml not found, using defaults")
            self.strategies = self.get_default_strategies()

        try:
            with open('prompts/examples.yaml', 'r') as f:
                self.examples = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("prompts/examples.yaml not found, no examples will be used.")
            self.examples = []
    
    def get_default_config(self) -> Dict:
        """Default configuration for prompt routing"""
        return {
            "user_types": {
                "founder": {
                    "default_strategy": "chain_of_thought",
                    "trust_threshold": 0.8
                },
                "customer": {
                    "default_strategy": "plain",
                    "trust_threshold": 0.7
                },
                "developer": {
                    "default_strategy": "react",
                    "trust_threshold": 0.6
                },
                "support": {
                    "default_strategy": "reflection",
                    "trust_threshold": 0.5
                }
            },
            "response_routing": {
                "auto_run_threshold": 0.95,
                "confirm_threshold": 0.70,
                "escalate_threshold": 0.0
            }
        }
    
    def get_default_strategies(self) -> Dict:
        """Default prompt strategies"""
        return {
            "plain": {
                "description": "Direct response without explicit reasoning",
                "template": "Answer the user's question directly and helpfully."
            },
            "cot": {
                "description": "Chain of Thought reasoning",
                "template": "Think step by step about the user's question, then provide a clear answer."
            },
            "react": {
                "description": "Reasoning and Acting pattern",
                "template": "Analyze the question, consider what information you need, then provide a reasoned response."
            },
            "reflect": {
                "description": "Reflection on response quality",
                "template": "Answer the question, then reflect on whether your response is helpful and complete."
            },
            "revise": {
                "description": "Revision of initial response",
                "template": "Provide an initial answer, then revise it to be more accurate and helpful."
            }
        }
    
    def format_examples(self) -> str:
        """Format few-shot examples for the prompt (disabled for token optimization)"""
        # Disabled examples to reduce prompt size - move to fine-tuning later
        return ""

    def get_system_prompt(self, user_type: str = "visitor") -> str:
        """Get optimized system prompt based on user type"""
        approach = 'technical' if user_type == 'developer' else 'clear' if user_type == 'customer' else 'simple'
        
        base_prompt = """You are Tuthand, a helpful AI assistant for websites.

RULES:
- Be professional, helpful, and transparent about limitations
- HIGH confidence (>80%): Give direct answers
- MEDIUM confidence (50-80%): Ask for clarification  
- LOW confidence (<50%): Escalate to human support
- Auto-escalate: politics, personal data, medical/legal advice

RESPONSE FORMAT:
Return JSON only: {{"response": "your answer", "confidence": 0.95}}
User type: {} (respond {})""".format(user_type, approach)

        return base_prompt
    
    def select_strategy(self, query: UserQuery) -> PromptStrategy:
        """Select appropriate prompt strategy based on query complexity and user type"""
        # Get complexity-based strategy from config
        complexity_factors = self.prompt_config.get("prompt_selection", {}).get("complexity_factors", {})
        
        # Check for sensitive topics first
        sensitive_keywords = complexity_factors.get("sensitive", {}).get("keywords", [])
        if any(keyword in query.text.lower() for keyword in sensitive_keywords):
            return PromptStrategy.ESCALATION if hasattr(PromptStrategy, 'ESCALATION') else PromptStrategy.REFLECT
        
        # Check for complex queries
        complex_keywords = complexity_factors.get("complex", {}).get("keywords", [])
        if any(keyword in query.text.lower() for keyword in complex_keywords):
            return PromptStrategy.REACT
        
        # Check for moderate complexity
        moderate_keywords = complexity_factors.get("moderate", {}).get("keywords", [])
        if any(keyword in query.text.lower() for keyword in moderate_keywords):
            return PromptStrategy.CHAIN_OF_THOUGHT
        
        # Check for simple queries
        simple_keywords = complexity_factors.get("simple", {}).get("keywords", [])
        if any(keyword in query.text.lower() for keyword in simple_keywords):
            return PromptStrategy.PLAIN
        
        # Fall back to user type default
        user_config = self.prompt_config.get("user_types", {}).get(query.user_type, {})
        default_strategy = user_config.get("default_strategy", "plain")
        
        # Map strategy names from config to enum
        strategy_mapping = {
            "chain_of_thought": PromptStrategy.CHAIN_OF_THOUGHT,
            "react": PromptStrategy.REACT,
            "reflection": PromptStrategy.REFLECT,
            "plain": PromptStrategy.PLAIN
        }
        
        return strategy_mapping.get(default_strategy, PromptStrategy.PLAIN)
    
    def get_cache_key(self, query: UserQuery) -> str:
        """Generate cache key for identical queries"""
        return "{}_{}".format(query.text.lower().strip(), query.user_type)
    
    def select_strategy_with_experiment(self, query: UserQuery) -> PromptStrategy:
        """Strategy selection with A/B testing support"""
        experiments = self.prompt_config.get("ab_testing", {}).get("experiments", {})
        
        # Check if user is in experiment
        if "strategy_comparison" in experiments:
            import random
            if random.random() < 0.5:  # 50% split
                return PromptStrategy.PLAIN
            else:
                return PromptStrategy.CHAIN_OF_THOUGHT
        
        return self.select_strategy(query)  # Default behavior
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 400) -> str:
        """Basic prompt compression for token optimization"""
        if len(prompt.split()) > max_tokens:
            # Simple compression: remove redundant phrases
            compressed = prompt.replace("Please respond according to the strategy above.", "")
            compressed = compressed.replace("Remember: Your goal is to be trustworthy and useful. When uncertain, be transparent about limitations.", "")
            return compressed.strip()
        return prompt
    
    def assess_query_complexity(self, query: str) -> float:
        """Assess query complexity for enhanced confidence calculation"""
        complexity_score = 0.0
        
        # Length-based complexity
        if len(query.split()) > 20:
            complexity_score += 0.3
        elif len(query.split()) > 10:
            complexity_score += 0.1
        
        # Question type complexity
        if any(word in query.lower() for word in ["how", "why", "explain", "compare"]):
            complexity_score += 0.2
        elif any(word in query.lower() for word in ["what", "when", "where"]):
            complexity_score += 0.1
        
        # Technical complexity
        if any(word in query.lower() for word in ["implement", "integrate", "api", "architecture"]):
            complexity_score += 0.3
        
        return min(complexity_score, 1.0)
    
    def build_prompt(self, query: UserQuery, strategy: PromptStrategy) -> str:
        """Build final prompt based on strategy"""
        system_prompt = self.get_system_prompt(query.user_type)
        strategy_config = self.strategies.get(strategy.value, {})
        strategy_instruction = strategy_config.get("template", "Answer the question helpfully.")
        examples = self.format_examples()
        
        prompt = f"""{system_prompt}
{examples}

STRATEGY: {strategy_instruction}

USER QUERY: {query.text}

Please respond according to the strategy above."""
        
        return prompt
    
    
    
    def determine_trust_level(self, confidence: float) -> TrustLevel:
        """Determine trust level based on confidence score"""
        thresholds = self.prompt_config.get("response_routing", {})
        
        if confidence >= thresholds.get("auto_run_threshold", 0.8):
            return TrustLevel.AUTO_RUN
        elif confidence >= thresholds.get("confirm_threshold", 0.5):
            return TrustLevel.CONFIRM
        else:
            return TrustLevel.ESCALATE
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage"""
        # GPT-4o-mini pricing (updated rates)
        input_cost_per_token = 0.15 / 1000  # $0.15 per 1k tokens
        output_cost_per_token = 0.6 / 1000  # $0.6 per 1k tokens
        
        # Rough estimate assuming 70% input, 30% output
        return (tokens * 0.7 * input_cost_per_token) + (tokens * 0.3 * output_cost_per_token)
    
    def log_response(self, query: UserQuery, response: AssistantResponse, strategy: PromptStrategy):
        """Log response to database for analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO responses (timestamp, user_type, query, response, trust_level, 
                                 tokens_used, response_time, cost_estimate, confidence_score, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            query.user_type,
            query.text,
            response.text,
            response.trust_level.value,
            response.metrics.tokens_used,
            response.metrics.response_time,
            response.metrics.cost_estimate,
            response.metrics.confidence_score,
            strategy.value
        ))
        
        conn.commit()
        conn.close()
    
    async def process_query(self, query: UserQuery, stream: bool = False) -> Union[AssistantResponse, AsyncGenerator]:
        """Process user query and return response with trust level, optionally streaming"""
        start_time = time.time()
        
        # Cache disabled for debugging
        # cache_key = self.get_cache_key(query)
        # if not stream and cache_key in self.response_cache:
        #     cached_response = self.response_cache[cache_key]
        #     logger.info("Cache hit for query: {}...".format(query.text[:30]))
        #     # Update response time for cached response
        #     cached_response.metrics.response_time = time.time() - start_time
        #     cached_response.reasoning += " (cached)"
        #     return cached_response
        
        # Select strategy
        strategy = self.select_strategy(query)
        logger.info("Selected strategy: {} for user type: {}".format(strategy.value, query.user_type))
        
        # Build and optimize prompt
        prompt = self.build_prompt(query, strategy)
        prompt = self.optimize_prompt(prompt)
        
        try:
            if stream:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    stream=True
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                response_content = response.choices[0].message.content
                
                try:
                    parsed_response = json.loads(response_content)
                    response_text = parsed_response.get("response", "Error: Could not parse response.")
                    confidence = parsed_response.get("confidence", 0.0)
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from LLM: %s", response_content)
                    response_text = "I apologize, but I received an unreadable response from the AI. Please try again."
                    confidence = 0.0
                
                tokens_used = response.usage.total_tokens
                response_time = time.time() - start_time
                
                # Calculate metrics
                trust_level = self.determine_trust_level(confidence)
                cost_estimate = self.estimate_cost(tokens_used)
                
                metrics = ResponseMetrics(
                    tokens_used=tokens_used,
                    response_time=response_time,
                    cost_estimate=cost_estimate,
                    confidence_score=confidence,
                    trust_level=trust_level
                )
                
                assistant_response = AssistantResponse(
                    text=response_text,
                    trust_level=trust_level,
                    metrics=metrics,
                    reasoning="Used {} strategy with {:.2f} confidence".format(strategy.value, confidence)
                )
                
                # Cache disabled for debugging
                # if assistant_response.trust_level == TrustLevel.AUTO_RUN:
                #     self.response_cache[cache_key] = assistant_response
                #     logger.info("Cached high-confidence response for: {}...".format(query.text[:30]))
                
                # Log for analytics
                self.log_response(query, assistant_response, strategy)
                
                return assistant_response
            
        except Exception as e:
            logger.error("Error processing query: %s", e)
            # Return fallback response
            return AssistantResponse(
                text="I apologize, but I'm having trouble processing your request right now. Please try again.",
                trust_level=TrustLevel.ESCALATE,
                metrics=ResponseMetrics(
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    cost_estimate=0.0,
                    confidence_score=0.0,
                    trust_level=TrustLevel.ESCALATE
                ),
                reasoning="Error occurred during processing"
            )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_responses,
                AVG(response_time) as avg_response_time,
                AVG(tokens_used) as avg_tokens,
                AVG(cost_estimate) as avg_cost,
                AVG(confidence_score) as avg_confidence
            FROM responses
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            "total_responses": stats[0] or 0,
            "avg_response_time": round(stats[1] or 0, 2),
            "avg_tokens": round(stats[2] or 0, 0),
            "avg_cost": round(stats[3] or 0, 4),
            "avg_confidence": round(stats[4] or 0, 2)
        }

def main():
    """Main function for testing the assistant"""
    assistant = TuthandAssistant()
    
    print("🤖 Tuthand Assistant - Week 2 Prototype")
    print("=" * 50)
    print("Choose mode:")
    print("1. Run demo queries")
    print("2. Interactive mode (ask your own questions)")
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    if mode == "1":
        # Demo queries
        test_queries = [
            UserQuery("What is Tuthand?", "customer"),
            UserQuery("How does pricing work?", "customer"),
            UserQuery("Can you explain the architecture?", "developer"),
            UserQuery("What's your favorite color?", "founder"),
        ]
        
        async def run_tests():
            for query in test_queries:
                print("\n💬 Query: {}".format(query.text))
                print("👤 User Type: {}".format(query.user_type))
                
                full_response_text = ""
                start_time = time.time()
                
                try:
                    stream = await assistant.process_query(query, stream=True)
                    
                    
                    full_response_content = ""
                    for chunk in stream:
                        full_response_content += chunk.choices[0].delta.content or ""


                    try:
                        parsed_response = json.loads(full_response_content)
                        response_text = parsed_response.get("response", "Error: Could not parse response.")
                        confidence = float(parsed_response.get("confidence", 0.0))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logger.error("Failed to decode JSON from LLM in demo: %s", full_response_content)
                        response_text = "I apologize, but I received an unreadable response in demo. Please try again."
                        confidence = 0.0

                    print(response_text)
                    
                    response_time = time.time() - start_time
                    # Better token estimation: roughly 1.3 tokens per word + prompt tokens
                    tokens_used = int(len(response_text.split()) * 1.3) + 200  # Add estimated prompt tokens
                    
                    trust_level = assistant.determine_trust_level(confidence)
                    cost_estimate = assistant.estimate_cost(tokens_used)
                    
                    assistant_response = AssistantResponse(
                        text=response_text,
                        trust_level=trust_level,
                        metrics=ResponseMetrics(
                            tokens_used=tokens_used,
                            response_time=response_time,
                            cost_estimate=cost_estimate,
                            confidence_score=confidence,
                            trust_level=trust_level
                        ),
                        reasoning="Used streaming with {:.2f} confidence".format(confidence)
                    )
                    assistant.log_response(query, assistant_response, assistant.select_strategy(query))

                    print(f"\n\n📊 METRICS:")
                    print("🔒 Trust Level: {}".format(trust_level.value))
                    print("📊 Confidence: {:.2f}".format(confidence))
                    print(f"⏱️ Response Time: {response_time:.2f}s")
                    print("💰 Cost: ${:.4f}".format(cost_estimate))
                    print("🧠 Reasoning: {}".format(assistant_response.reasoning))

                except Exception as e:
                    logger.error("Error during streaming demo: %s", e)
                    print("\nI apologize, but I'm having trouble processing this demo request right now.")

                print("-" * 50)
        
        asyncio.run(run_tests())
    
    elif mode == "2":
        # Interactive mode
        print("\n🎯 Interactive Mode")
        print("User types: founder, customer, developer, support")
        print("Type 'quit' to exit, 'stats' for performance stats")
        print("-" * 50)
        
        async def interactive_session():
            while True:
                print("\n" + "="*50)
                query_text = input("💬 Your question: ").strip()
                
                if query_text.lower() == 'quit':
                    break
                elif query_text.lower() == 'stats':
                    print("\n📊 Performance Statistics (Last 24h):")
                    stats = assistant.get_performance_stats()
                    for key, value in stats.items():
                        print("{}: {}".format(key, value))
                    continue
                elif not query_text:
                    continue
                
                user_type = input("👤 User type [founder/customer/developer/support]: ").strip()
                if user_type not in ['founder', 'customer', 'developer', 'support']:
                    user_type = 'customer'
                
                query = UserQuery(query_text, user_type)
                
                print(f"\n🔄 Processing...")
                
                full_response_content = ""
                start_time = time.time()
                
                try:
                    stream = await assistant.process_query(query, stream=True)
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_response_content += content
                        

                    try:
                        parsed_response = json.loads(full_response_content)
                        response_text = parsed_response.get("response", "Error: Could not parse response.")
                        confidence = float(parsed_response.get("confidence", 0.0))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logger.error("Failed to decode JSON from LLM: %s", full_response_content)
                        response_text = "I apologize, but I received an unreadable response. Please try again."
                        confidence = 0.0

                    print(response_text)

                    # Once streaming is complete, get the full response object for metrics
                    response_time = time.time() - start_time
                    # Better token estimation: roughly 1.3 tokens per word + prompt tokens
                    tokens_used = int(len(response_text.split()) * 1.3) + 200  # Add estimated prompt tokens
                    
                    trust_level = assistant.determine_trust_level(confidence)
                    cost_estimate = assistant.estimate_cost(tokens_used)
                    
                    metrics = ResponseMetrics(
                        tokens_used=tokens_used,
                        response_time=response_time,
                        cost_estimate=cost_estimate,
                        confidence_score=confidence,
                        trust_level=trust_level
                    )
                    
                    assistant_response = AssistantResponse(
                        text=response_text,
                        trust_level=trust_level,
                        metrics=metrics,
                        reasoning="Used streaming with {:.2f} confidence".format(confidence)
                    )
                    
                    assistant.log_response(query, assistant_response, assistant.select_strategy(query))

                    print(f"\n\n📊 METRICS:")
                    print("🔒 Trust Level: {}".format(trust_level.value))
                    print("📊 Confidence: {:.2f}".format(confidence))
                    print(f"⏱️ Response Time: {response_time:.2f}s")
                    print("💰 Cost: ${:.4f}".format(cost_estimate))

                except Exception as e:
                    logger.error("Error during streaming: %s", e)
                    print("\nI apologize, but I'm having trouble processing your request right now.")

                print("\n" + "-" * 50)
        
        asyncio.run(interactive_session())
    
    else:
        print("Invalid mode. Please run again and choose 1 or 2.")
        return
    
    # Show final performance stats
    print("\n📊 Final Performance Statistics (Last 24h):")
    stats = assistant.get_performance_stats()
    for key, value in stats.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":
    main()