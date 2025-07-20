#!/usr/bin/env python3
"""
Enhanced Retrieval System - Week 3 Implementation
Context-aware memory retrieval with hybrid search and reranking
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from .vector_store import VectorMemoryStore

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with relevance scoring"""
    content: str
    source: str
    confidence: float
    relevance_score: float
    metadata: Dict[str, Any]
    retrieval_reason: str

class EnhancedRetrieval:
    """Advanced retrieval system with context awareness and hybrid search"""
    
    def __init__(self, vector_store: VectorMemoryStore):
        self.vector_store = vector_store
        self.context_cache = {}
        self.relevance_weights = {
            "semantic_similarity": 0.4,
            "temporal_relevance": 0.2,
            "context_match": 0.2,
            "confidence_score": 0.1,
            "source_authority": 0.1
        }
    
    def retrieve_context(self, 
                        query: str,
                        user_context: Optional[Dict] = None,
                        conversation_history: Optional[List[Dict]] = None,
                        top_k: int = 5,
                        enable_reranking: bool = True) -> List[RetrievalResult]:
        """
        Retrieve relevant context with advanced filtering and ranking
        """
        # Expand query with context
        expanded_query = self._expand_query(query, user_context, conversation_history)
        
        # Perform hybrid search
        search_results = self._hybrid_search(expanded_query, user_context, top_k * 2)
        
        # Apply context filtering
        filtered_results = self._context_filter(search_results, user_context, conversation_history)
        
        # Rerank results if enabled
        if enable_reranking:
            reranked_results = self._rerank_results(query, filtered_results, user_context)
        else:
            reranked_results = filtered_results
        
        # Convert to RetrievalResult objects
        enhanced_results = []
        for result in reranked_results[:top_k]:
            enhanced_results.append(RetrievalResult(
                content=result["content"],
                source=result["metadata"].get("source", "unknown"),
                confidence=result["metadata"].get("confidence", 0.5),
                relevance_score=result.get("final_score", result["score"]),
                metadata=result["metadata"],
                retrieval_reason=result.get("reason", "semantic_similarity")
            ))
        
        return enhanced_results
    
    def _expand_query(self, 
                     query: str, 
                     user_context: Optional[Dict],
                     conversation_history: Optional[List[Dict]]) -> str:
        """Expand query with contextual information"""
        expanded_parts = [query]
        
        # Add user type context
        if user_context and user_context.get("user_type"):
            user_type = user_context["user_type"]
            if user_type == "developer":
                expanded_parts.append("technical implementation architecture")
            elif user_type == "founder":
                expanded_parts.append("business strategy overview")
            elif user_type == "customer":
                expanded_parts.append("user-friendly explanation")
        
        # Add recent conversation topics
        if conversation_history:
            recent_topics = self._extract_topics(conversation_history[-3:])
            expanded_parts.extend(recent_topics)
        
        return " ".join(expanded_parts)
    
    def _extract_topics(self, conversation_history: List[Dict]) -> List[str]:
        """Extract key topics from recent conversation"""
        topics = []
        for exchange in conversation_history:
            query = exchange.get("query", "")
            # Simple keyword extraction
            keywords = re.findall(r'\b(?:pricing|features|api|integration|support|documentation)\b', 
                                query.lower())
            topics.extend(keywords)
        return list(set(topics))
    
    def _hybrid_search(self, 
                      query: str, 
                      user_context: Optional[Dict],
                      top_k: int) -> List[Dict]:
        """Combine vector similarity with keyword and context matching"""
        
        # Vector similarity search
        vector_results = self.vector_store.search_memories(
            query=query,
            top_k=top_k,
            filter_metadata=self._build_metadata_filter(user_context)
        )
        
        # Keyword matching boost
        keyword_boosted = self._apply_keyword_boost(query, vector_results)
        
        # Temporal relevance boost
        temporal_boosted = self._apply_temporal_boost(keyword_boosted)
        
        return temporal_boosted
    
    def _build_metadata_filter(self, user_context: Optional[Dict]) -> Optional[Dict]:
        """Build metadata filter based on user context"""
        if not user_context:
            return None
        
        filter_dict = {}
        
        # Filter by user type if specified
        if user_context.get("user_type"):
            # Don't filter by user_type - allow cross-type learning
            pass
        
        # Filter by minimum confidence
        confidence_threshold = user_context.get("min_confidence", 0.3)
        if confidence_threshold > 0:
            filter_dict["confidence"] = {"$gte": confidence_threshold}
        
        return filter_dict if filter_dict else None
    
    def _apply_keyword_boost(self, query: str, results: List[Dict]) -> List[Dict]:
        """Boost results that contain exact keyword matches"""
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result["content"].lower().split())
            keyword_overlap = len(query_words.intersection(content_words))
            
            if keyword_overlap > 0:
                boost_factor = 1 + (keyword_overlap * 0.1)
                result["score"] *= boost_factor
                result["keyword_boost"] = boost_factor
        
        return results
    
    def _apply_temporal_boost(self, results: List[Dict]) -> List[Dict]:
        """Boost more recent memories"""
        now = datetime.now()
        
        for result in results:
            timestamp_str = result["metadata"].get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_days = (now - timestamp).days
                
                # Boost factor decreases with age
                if age_days < 1:
                    boost_factor = 1.2
                elif age_days < 7:
                    boost_factor = 1.1
                elif age_days < 30:
                    boost_factor = 1.0
                else:
                    boost_factor = 0.9
                
                result["score"] *= boost_factor
                result["temporal_boost"] = boost_factor
                
            except (ValueError, TypeError):
                # If timestamp parsing fails, no temporal boost
                result["temporal_boost"] = 1.0
        
        return results
    
    def _context_filter(self, 
                       results: List[Dict],
                       user_context: Optional[Dict],
                       conversation_history: Optional[List[Dict]]) -> List[Dict]:
        """Filter results based on conversation context"""
        if not conversation_history:
            return results
        
        # If conversation history is very short, be more permissive
        if len(conversation_history) < 3:
            # For short conversations, just apply a lower score threshold
            filtered_results = []
            for result in results:
                if result["score"] > 0.3:  # Lower threshold for short conversations
                    result["context_score"] = 0.5
                    result["topic_match"] = False
                    filtered_results.append(result)
            return filtered_results
        
        # Get recent conversation topics
        recent_topics = self._extract_topics(conversation_history[-5:])
        
        filtered_results = []
        for result in results:
            content = result["content"].lower()
            
            # Check topic relevance
            topic_match = any(topic in content for topic in recent_topics)
            
            # Check for context continuity
            context_score = self._calculate_context_score(result, conversation_history)
            
            # Apply filters - Made less strict for better memory recall
            if topic_match or context_score > 0.3 or result["score"] > 0.5:
                result["context_score"] = context_score
                result["topic_match"] = topic_match
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_context_score(self, 
                               result: Dict, 
                               conversation_history: List[Dict]) -> float:
        """Calculate how well this result fits the conversation context"""
        if not conversation_history:
            return 0.5
        
        content = result["content"].lower()
        context_score = 0.0
        
        # Check overlap with recent exchanges
        for exchange in conversation_history[-3:]:
            query = exchange.get("query", "").lower()
            response = exchange.get("response", "").lower()
            
            # Word overlap scoring
            query_words = set(query.split())
            response_words = set(response.split())
            content_words = set(content.split())
            
            query_overlap = len(query_words.intersection(content_words))
            response_overlap = len(response_words.intersection(content_words))
            
            context_score += (query_overlap + response_overlap) * 0.1
        
        return min(context_score, 1.0)
    
    def _rerank_results(self, 
                       original_query: str,
                       results: List[Dict],
                       user_context: Optional[Dict]) -> List[Dict]:
        """Advanced reranking with multiple signals"""
        
        for result in results:
            final_score = 0.0
            
            # Semantic similarity (base score)
            semantic_score = result["score"]
            final_score += semantic_score * self.relevance_weights["semantic_similarity"]
            
            # Temporal relevance
            temporal_boost = result.get("temporal_boost", 1.0)
            final_score += (temporal_boost - 1.0) * self.relevance_weights["temporal_relevance"]
            
            # Context match
            context_score = result.get("context_score", 0.0)
            final_score += context_score * self.relevance_weights["context_match"]
            
            # Confidence score
            confidence = result["metadata"].get("confidence", 0.5)
            final_score += confidence * self.relevance_weights["confidence_score"]
            
            # Source authority
            source_authority = self._get_source_authority(result["metadata"].get("source", ""))
            final_score += source_authority * self.relevance_weights["source_authority"]
            
            result["final_score"] = final_score
            result["ranking_breakdown"] = {
                "semantic": semantic_score,
                "temporal": temporal_boost,
                "context": context_score,
                "confidence": confidence,
                "source_authority": source_authority
            }
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
    def _get_source_authority(self, source: str) -> float:
        """Get authority score for different sources"""
        source_scores = {
            "official_docs": 1.0,
            "faq": 0.9,
            "support_knowledge": 0.8,
            "user_interaction": 0.6,
            "external_content": 0.4,
            "unknown": 0.5
        }
        return source_scores.get(source, 0.5)
    
    def cache_context(self, session_id: str, context: Dict):
        """Cache context for session-based optimization"""
        self.context_cache[session_id] = {
            "context": context,
            "timestamp": datetime.now()
        }
        
        # Clean old cache entries
        self._clean_context_cache()
    
    def _clean_context_cache(self):
        """Remove old cache entries"""
        cutoff = datetime.now() - timedelta(hours=1)
        self.context_cache = {
            k: v for k, v in self.context_cache.items()
            if v["timestamp"] > cutoff
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            "cache_size": len(self.context_cache),
            "relevance_weights": self.relevance_weights,
            "vector_store_stats": self.vector_store.get_memory_stats()
        }

# Factory function
def create_enhanced_retrieval(vector_store: VectorMemoryStore) -> EnhancedRetrieval:
    """Create an enhanced retrieval system"""
    return EnhancedRetrieval(vector_store)