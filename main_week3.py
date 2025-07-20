#!/usr/bin/env python3
"""
Tuthand - Week 3 Prototype with Memory Systems and Enhanced Intelligence
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

# Week 3: Memory system imports
try:
    from memory.vector_store import create_vector_store
    from memory.memory_manager import create_memory_manager
    MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory system not available: {e}")
    MEMORY_AVAILABLE = False

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
    memory_context_used: bool = False
    retrieval_results: int = 0

@dataclass
class UserQuery:
    """User query with metadata"""
    text: str
    user_type: str = "customer"
    context: Optional[str] = None
    timestamp: datetime = None
    session_id: Optional[str] = None

@dataclass
class AssistantResponse:
    """Assistant response with metadata"""
    text: str
    trust_level: TrustLevel
    metrics: ResponseMetrics
    reasoning: Optional[str] = None
    memory_context: Optional[List[Dict]] = None

class PromptStrategy(Enum):
    """Available prompt strategies"""
    PLAIN = "plain"
    CHAIN_OF_THOUGHT = "cot"
    REACT = "react"
    REFLECT = "reflect"
    REVISE = "revise"
    MEMORY_ENHANCED = "memory_enhanced"  # New Week 3 strategy

class TuthandAssistant:
    """Main Tuthand AI Assistant with Week 3 Memory Features"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.db_path = "performance_metrics.db"
        self.response_cache = {}
        
        # Week 3: Memory system initialization
        if MEMORY_AVAILABLE:
            # Choose vector store provider (fallback to in-memory if others not available)
            vector_provider = os.getenv("VECTOR_STORE_PROVIDER", "inmemory")
            self.vector_store = create_vector_store(provider=vector_provider)
            self.memory_manager = create_memory_manager(self.vector_store)
            logger.info(f"Memory system initialized with {vector_provider} vector store")
        else:
            self.vector_store = None
            self.memory_manager = None
            logger.warning("Memory system disabled - running in Week 2 compatibility mode")
        
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
                strategy TEXT,
                session_id TEXT,
                memory_context_used BOOLEAN,
                retrieval_results INTEGER
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
                    "default_strategy": "memory_enhanced",
                    "trust_threshold": 0.8
                },
                "customer": {
                    "default_strategy": "memory_enhanced",
                    "trust_threshold": 0.7
                },
                "developer": {
                    "default_strategy": "memory_enhanced",
                    "trust_threshold": 0.6
                },
                "support": {
                    "default_strategy": "memory_enhanced",
                    "trust_threshold": 0.5
                }
            },
            "response_routing": {
                "auto_run_threshold": 0.95,
                "confirm_threshold": 0.70,
                "escalate_threshold": 0.0
            },
            "memory_settings": {
                "enable_memory": True,
                "context_limit": 5,
                "min_relevance_score": 0.3
            }
        }
    
    def get_default_strategies(self) -> Dict:
        """Default prompt strategies including memory-enhanced"""
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
            },
            "memory_enhanced": {
                "description": "Use relevant context from memory to enhance response",
                "template": "Using the provided context and conversation history, give a personalized and contextually relevant answer."
            }
        }
    
    def format_examples(self) -> str:
        """Format few-shot examples for the prompt (disabled for token optimization)"""
        # Disabled examples to reduce prompt size - move to fine-tuning later
        return ""

    def get_system_prompt(self, user_type: str = "visitor", memory_context: Optional[List[Dict]] = None) -> str:
        """Get optimized system prompt based on user type and memory context"""
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
        
        # Add memory context if available
        if memory_context and len(memory_context) > 0:
            context_section = "\n\nRELEVANT CONTEXT FROM CONVERSATION HISTORY:"
            
            # Prioritize correction notices and recent information
            correction_contexts = []
            regular_contexts = []
            conflicting_info = []
            
            for ctx in memory_context[:10]:  # Check more contexts for corrections
                if "CORRECTION:" in ctx['content'] or "UPDATE:" in ctx['content']:
                    correction_contexts.append(ctx)
                else:
                    regular_contexts.append(ctx)
            
            # Detect conflicting information patterns
            has_conflicts = self._detect_context_conflicts(memory_context)
            
            # Show correction notices first, then most recent regular contexts
            all_contexts = correction_contexts + regular_contexts
            
            # Always show correction notices, then fill remaining slots
            contexts_to_show = []
            correction_count = len(correction_contexts)
            
            # Add all correction notices first
            for ctx in correction_contexts:
                contexts_to_show.append(ctx)
            
            # Add regular contexts to fill up to 3 total, prioritizing recent ones
            remaining_slots = max(1, 3 - correction_count)  # At least 1 regular context
            for ctx in regular_contexts[:remaining_slots]:
                contexts_to_show.append(ctx)
            
            for i, ctx in enumerate(contexts_to_show):
                marker = "🔧 CORRECTION" if "CORRECTION:" in ctx['content'] or "UPDATE:" in ctx['content'] else ""
                context_section += f"\n{i+1}. {marker} {ctx['content']} (confidence: {ctx['confidence']:.2f})"
            
            if has_conflicts:
                context_section += "\n\n⚠️ CONFLICT DETECTED: Multiple conflicting pieces of information found. If you cannot determine the correct answer with high confidence, set confidence to 0.3 to escalate to human support."
            
            context_section += "\n\nCRITICAL INSTRUCTIONS:\n1. CORRECTION and UPDATE entries contain the LATEST, MOST ACCURATE information\n2. When CORRECTION notices are present, IGNORE conflicting older information\n3. ALWAYS prioritize user corrections over previous statements\n4. If there are conflicts, explicitly acknowledge the correction: 'I see you've corrected that - your [information] is [corrected value]'\n5. If conflicting information makes you uncertain, lower your confidence to trigger escalation"
            base_prompt += context_section
        
        return base_prompt
    
    def select_strategy(self, query: UserQuery) -> PromptStrategy:
        """Select appropriate prompt strategy based on query complexity and user type"""
        # Week 3: Default to memory-enhanced strategy if memory is available
        if MEMORY_AVAILABLE and self.memory_manager:
            return PromptStrategy.MEMORY_ENHANCED
        
        # Fall back to Week 2 strategy selection
        user_config = self.prompt_config.get("user_types", {}).get(query.user_type, {})
        default_strategy = user_config.get("default_strategy", "plain")
        
        strategy_mapping = {
            "chain_of_thought": PromptStrategy.CHAIN_OF_THOUGHT,
            "react": PromptStrategy.REACT,
            "reflection": PromptStrategy.REFLECT,
            "plain": PromptStrategy.PLAIN,
            "memory_enhanced": PromptStrategy.MEMORY_ENHANCED
        }
        
        return strategy_mapping.get(default_strategy, PromptStrategy.PLAIN)
    
    def get_cache_key(self, query: UserQuery) -> str:
        """Generate cache key for identical queries"""
        return "{}_{}".format(query.text.lower().strip(), query.user_type)
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 400) -> str:
        """Basic prompt compression for token optimization"""
        if len(prompt.split()) > max_tokens:
            # Simple compression: remove redundant phrases
            compressed = prompt.replace("Please respond according to the strategy above.", "")
            return compressed.strip()
        return prompt
    
    def build_prompt(self, query: UserQuery, strategy: PromptStrategy, memory_context: Optional[List[Dict]] = None) -> str:
        """Build final prompt based on strategy and memory context"""
        system_prompt = self.get_system_prompt(query.user_type, memory_context)
        strategy_config = self.strategies.get(strategy.value, {})
        strategy_instruction = strategy_config.get("template", "Answer the question helpfully.")
        examples = self.format_examples()
        
        prompt = f"""{system_prompt}
{examples}

STRATEGY: {strategy_instruction}

USER QUERY: {query.text}

Please respond according to the strategy above."""
        
        return prompt
    
    def _get_dynamic_top_k(self, query: str) -> int:
        """Get dynamic top_k based on query type"""
        query_lower = query.lower()
        
        # Personal information queries need deep search
        if any(phrase in query_lower for phrase in ["my name", "who am i", "about me", "i am", "i'm"]):
            return 10
        
        # Historical/memory queries need more context
        elif any(phrase in query_lower for phrase in ["earlier", "previous", "we discussed", "talked about", 
                                                       "what did", "remind me", "you said", "you mentioned"]):
            return 8
        
        # Specific information queries might need more context
        elif any(phrase in query_lower for phrase in ["tell me more", "explain", "details about", "what about"]):
            return 5
        
        # Default for general queries
        else:
            return 3
    
    def _calculate_importance_score(self, content: str) -> float:
        """Calculate importance multiplier based on content type"""
        import re
        content_lower = content.lower()
        
        # Personal information patterns (highest priority)
        personal_patterns = [
            r"my name is (\w+)",
            r"i(?:'m| am) (\w+)",
            r"call me (\w+)",
            r"i work at",
            r"i'm a",
            r"i am a",
            r"i live in",
            r"my email",
            r"my phone"
        ]
        
        # Requirements and business info patterns (high priority)
        business_patterns = [
            r"we need",
            r"looking for",
            r"we want",
            r"our goal",
            r"we have \d+ (customers|users|visitors|employees)",
            r"our company",
            r"our budget",
            r"we're building"
        ]
        
        # Context questions (medium-high priority)
        context_patterns = [
            r"what.*my name",
            r"what.*we.*discuss",
            r"what.*talk.*about",
            r"remind me",
            r"earlier you said",
            r"you mentioned"
        ]
        
        # Check for personal information (highest priority)
        for pattern in personal_patterns:
            if re.search(pattern, content_lower):
                return 2.5  # Very high importance
        
        # Check for business requirements
        for pattern in business_patterns:
            if re.search(pattern, content_lower):
                return 2.0  # High importance
        
        # Check for context questions
        for pattern in context_patterns:
            if re.search(pattern, content_lower):
                return 1.5  # Medium-high importance
        
        # Product/technical information
        if any(word in content_lower for word in ["tuthand", "api", "integration", "features", "pricing"]):
            return 1.3  # Medium importance
        
        # Default importance
        return 1.0
    
    def _detect_context_conflicts(self, memory_context: List[Dict]) -> bool:
        """Detect conflicting information in retrieved context"""
        import re
        
        # Extract personal information from all contexts
        personal_info = {}
        
        for ctx in memory_context:
            content_lower = ctx['content'].lower()
            
            # Check for name conflicts
            name_patterns = [
                r"my name is (\w+)",
                r"i(?:'m| am) (\w+)",
                r"call me (\w+)"
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, content_lower)
                for match in matches:
                    if 'name' not in personal_info:
                        personal_info['name'] = []
                    personal_info['name'].append(match.lower())
            
            # Check for company conflicts
            company_patterns = [
                r"i work at ([^.]+)",
                r"our company is ([^.]+)",
                r"we are ([^.]+)"
            ]
            
            for pattern in company_patterns:
                matches = re.findall(pattern, content_lower)
                for match in matches:
                    if 'company' not in personal_info:
                        personal_info['company'] = []
                    personal_info['company'].append(match.strip().lower())
        
        # Check for conflicts in each category
        for category, values in personal_info.items():
            if len(set(values)) > 1:  # Multiple different values
                logger.info(f"Conflict detected in {category}: {values}")
                return True
        
        return False
    
    def _calculate_recency_boost(self, timestamp_str: str) -> float:
        """Calculate recency boost based on how recent the memory is"""
        try:
            from datetime import datetime
            
            # Parse timestamp
            if timestamp_str:
                # Handle different timestamp formats
                if 'T' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            else:
                return 1.0  # No timestamp info
            
            # Calculate age in minutes
            now = datetime.now()
            age_minutes = (now - timestamp).total_seconds() / 60
            
            # Apply recency boost
            if age_minutes < 5:      # Very recent (last 5 minutes)
                return 1.8
            elif age_minutes < 30:   # Recent (last 30 minutes) 
                return 1.5
            elif age_minutes < 120:  # Moderately recent (last 2 hours)
                return 1.3
            elif age_minutes < 1440: # Within day (last 24 hours)
                return 1.1
            else:                    # Older than 1 day
                return 0.9
                
        except (ValueError, TypeError) as e:
            # If timestamp parsing fails, no boost
            return 1.0
    
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
                                 tokens_used, response_time, cost_estimate, confidence_score, strategy,
                                 session_id, memory_context_used, retrieval_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            strategy.value,
            query.session_id,
            response.metrics.memory_context_used,
            response.metrics.retrieval_results
        ))
        
        conn.commit()
        conn.close()
    
    async def process_query(self, query: UserQuery, stream: bool = False) -> Union[AssistantResponse, AsyncGenerator]:
        """Process user query with memory enhancement"""
        start_time = time.time()
        
        # Week 3: Retrieve relevant memory context
        memory_context = []
        memory_context_used = False
        retrieval_results = 0
        
        if MEMORY_AVAILABLE and self.memory_manager and query.session_id:
            try:
                # Dynamic top_k based on query type
                top_k = self._get_dynamic_top_k(query.text)
                
                # TEMPORARY FIX: Use direct vector search instead of enhanced retrieval
                direct_results = self.vector_store.search_memories(
                    query=query.text,
                    top_k=top_k,
                    min_confidence=0.3
                )
                
                # Apply importance scoring and convert to memory context format
                memory_context = []
                scored_results = []
                
                # Calculate importance and recency adjusted scores
                for result in direct_results:
                    importance = self._calculate_importance_score(result["content"])
                    
                    # Get timestamp from metadata
                    timestamp = result["metadata"].get("timestamp", "")
                    recency_boost = self._calculate_recency_boost(timestamp)
                    
                    # Combine importance and recency
                    combined_multiplier = importance * recency_boost
                    adjusted_score = result["score"] * combined_multiplier
                    
                    scored_results.append({
                        "original_result": result,
                        "importance": importance,
                        "recency_boost": recency_boost,
                        "combined_multiplier": combined_multiplier,
                        "adjusted_score": adjusted_score
                    })
                
                # Sort by adjusted score
                scored_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
                
                # Filter and format results
                for scored in scored_results:
                    result = scored["original_result"]
                    if scored["adjusted_score"] > 0.2:  # Use adjusted score for threshold
                        memory_context.append({
                            "content": result["content"],
                            "source": result["metadata"].get("source", "unknown"),
                            "confidence": result["metadata"].get("confidence", 0.5),
                            "relevance_score": result["score"],
                            "importance_score": scored["importance"],
                            "recency_boost": scored["recency_boost"],
                            "combined_multiplier": scored["combined_multiplier"],
                            "adjusted_score": scored["adjusted_score"],
                            "retrieval_reason": f"importance_recency_weighted (importance: {scored['importance']:.1f}x, recency: {scored['recency_boost']:.1f}x)"
                        })
                
                memory_context_used = len(memory_context) > 0
                retrieval_results = len(memory_context)
                
                if memory_context_used:
                    logger.info(f"Retrieved {retrieval_results} memory contexts for query (direct search)")
                
            except Exception as e:
                logger.error(f"Memory retrieval failed: {e}")
                memory_context = []
        
        # Select strategy
        strategy = self.select_strategy(query)
        logger.info("Selected strategy: {} for user type: {}".format(strategy.value, query.user_type))
        
        # Build and optimize prompt
        prompt = self.build_prompt(query, strategy, memory_context)
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
                    trust_level=trust_level,
                    memory_context_used=memory_context_used,
                    retrieval_results=retrieval_results
                )
                
                assistant_response = AssistantResponse(
                    text=response_text,
                    trust_level=trust_level,
                    metrics=metrics,
                    reasoning="Used {} strategy with {:.2f} confidence{}".format(
                        strategy.value, 
                        confidence,
                        f", {retrieval_results} memory contexts" if memory_context_used else ""
                    ),
                    memory_context=memory_context if memory_context_used else None
                )
                
                # Week 3: Store interaction in memory
                if MEMORY_AVAILABLE and self.memory_manager and query.session_id:
                    try:
                        self.memory_manager.add_conversation_exchange(
                            session_id=query.session_id,
                            query=query.text,
                            response=response_text,
                            user_type=query.user_type,
                            trust_level=trust_level.value,
                            confidence=confidence,
                            strategy_used=strategy.value
                        )
                    except Exception as e:
                        logger.error(f"Failed to store interaction in memory: {e}")
                
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
                    trust_level=TrustLevel.ESCALATE,
                    memory_context_used=False,
                    retrieval_results=0
                ),
                reasoning="Error occurred during processing"
            )
    
    def create_session(self, user_type: str = "customer") -> str:
        """Create a new user session for memory tracking"""
        if MEMORY_AVAILABLE and self.memory_manager:
            return self.memory_manager.create_session(user_type)
        else:
            # Fallback session ID for Week 2 compatibility
            import uuid
            return str(uuid.uuid4())
    
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
                AVG(confidence_score) as avg_confidence,
                AVG(CASE WHEN memory_context_used = 1 THEN 1 ELSE 0 END) as memory_usage_rate,
                AVG(retrieval_results) as avg_retrieval_results
            FROM responses
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        base_stats = {
            "total_responses": stats[0] or 0,
            "avg_response_time": round(stats[1] or 0, 2),
            "avg_tokens": round(stats[2] or 0, 0),
            "avg_cost": round(stats[3] or 0, 4),
            "avg_confidence": round(stats[4] or 0, 2),
            "memory_usage_rate": round(stats[5] or 0, 2),
            "avg_retrieval_results": round(stats[6] or 0, 1)
        }
        
        # Add memory system stats if available
        if MEMORY_AVAILABLE and self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            base_stats.update({"memory_system": memory_stats})
        
        return base_stats

def main():
    """Main function for testing the Week 3 assistant"""
    assistant = TuthandAssistant()
    
    print("🤖 Tuthand Assistant - Week 3 Prototype with Memory")
    print("=" * 60)
    print("Features: Memory systems, context awareness, conversation tracking")
    print("Choose mode:")
    print("1. Run demo queries")
    print("2. Interactive mode (ask your own questions)")
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    if mode == "1":
        # Demo with session tracking
        print("\n🧠 Creating user session for memory tracking...")
        session_id = assistant.create_session("customer")
        print(f"Session ID: {session_id}")
        
        test_queries = [
            UserQuery("What is Tuthand?", "customer", session_id=session_id),
            UserQuery("How does pricing work?", "customer", session_id=session_id),
            UserQuery("Can you tell me more about the pricing you mentioned?", "customer", session_id=session_id),
            UserQuery("What features does it have?", "customer", session_id=session_id),
        ]
        
        async def run_tests():
            for i, query in enumerate(test_queries):
                print(f"\n💬 Query {i+1}: {query.text}")
                print("👤 User Type: {}".format(query.user_type))
                
                try:
                    response = await assistant.process_query(query, stream=False)
                    
                    print(f"🤖 Response: {response.text}")
                    print(f"\n📊 METRICS:")
                    print("🔒 Trust Level: {}".format(response.trust_level.value))
                    print("📊 Confidence: {:.2f}".format(response.metrics.confidence_score))
                    print(f"⏱️ Response Time: {response.metrics.response_time:.2f}s")
                    print("💰 Cost: ${:.4f}".format(response.metrics.cost_estimate))
                    print("🧠 Memory Used: {}".format("Yes" if response.metrics.memory_context_used else "No"))
                    if response.metrics.memory_context_used:
                        print(f"📚 Retrieved Contexts: {response.metrics.retrieval_results}")
                    print("🧠 Reasoning: {}".format(response.reasoning))

                except Exception as e:
                    logger.error("Error during demo: %s", e)
                    print("\nI apologize, but I'm having trouble processing this demo request right now.")

                print("-" * 60)
        
        asyncio.run(run_tests())
    
    elif mode == "2":
        # Interactive mode with session
        print("\n🎯 Interactive Mode with Memory")
        print("User types: founder, customer, developer, support")
        print("Type 'quit' to exit, 'stats' for performance stats, 'memory' for memory stats")
        print("-" * 60)
        
        # Create session
        user_type = input("👤 Your user type [founder/customer/developer/support]: ").strip()
        if user_type not in ['founder', 'customer', 'developer', 'support']:
            user_type = 'customer'
        
        session_id = assistant.create_session(user_type)
        print(f"🧠 Created session: {session_id}")
        
        async def interactive_session():
            while True:
                print("\n" + "="*60)
                query_text = input("💬 Your question: ").strip()
                
                if query_text.lower() == 'quit':
                    break
                elif query_text.lower() == 'stats':
                    print("\n📊 Performance Statistics (Last 24h):")
                    stats = assistant.get_performance_stats()
                    for key, value in stats.items():
                        if key != "memory_system":
                            print("{}: {}".format(key, value))
                    continue
                elif query_text.lower() == 'memory':
                    if MEMORY_AVAILABLE and assistant.memory_manager:
                        print("\n🧠 Memory System Statistics:")
                        memory_stats = assistant.memory_manager.get_memory_stats()
                        for key, value in memory_stats.items():
                            print("{}: {}".format(key, value))
                    else:
                        print("Memory system not available")
                    continue
                elif not query_text:
                    continue
                
                query = UserQuery(query_text, user_type, session_id=session_id)
                
                print(f"\n🔄 Processing with memory context...")
                
                try:
                    response = await assistant.process_query(query, stream=False)
                    
                    print(f"🤖 {response.text}")

                    print(f"\n📊 METRICS:")
                    print("🔒 Trust Level: {}".format(response.trust_level.value))
                    print("📊 Confidence: {:.2f}".format(response.metrics.confidence_score))
                    print(f"⏱️ Response Time: {response.metrics.response_time:.2f}s")
                    print("💰 Cost: ${:.4f}".format(response.metrics.cost_estimate))
                    print("🧠 Memory Used: {}".format("Yes" if response.metrics.memory_context_used else "No"))
                    if response.metrics.memory_context_used:
                        print(f"📚 Retrieved Contexts: {response.metrics.retrieval_results}")

                except Exception as e:
                    logger.error("Error during interaction: %s", e)
                    print("\nI apologize, but I'm having trouble processing your request right now.")

                print("\n" + "-" * 60)
        
        asyncio.run(interactive_session())
    
    else:
        print("Invalid mode. Please run again and choose 1 or 2.")
        return
    
    # Show final performance stats
    print("\n📊 Final Performance Statistics (Last 24h):")
    stats = assistant.get_performance_stats()
    for key, value in stats.items():
        if key != "memory_system":
            print("{}: {}".format(key, value))

if __name__ == "__main__":
    main()