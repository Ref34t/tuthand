#!/usr/bin/env python3
"""
Memory Manager - Week 3 Implementation
Manages short-term and long-term memory with conversation tracking
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import re
import numpy as np
from collections import defaultdict

from .vector_store import VectorMemoryStore
from .retrieval import EnhancedRetrieval

logger = logging.getLogger(__name__)

@dataclass
class ConversationExchange:
    """Single conversation exchange"""
    query: str
    response: str
    user_type: str
    trust_level: str
    confidence: float
    timestamp: datetime
    strategy_used: str

@dataclass
class UserContext:
    """User context and preferences"""
    user_type: str
    session_id: str
    confidence_history: List[float]
    topic_preferences: List[str]
    trust_level_history: List[str]
    last_interaction: datetime
    total_interactions: int

@dataclass
class SessionMemory:
    """Short-term session memory"""
    session_id: str
    user_context: UserContext
    conversation_history: List[ConversationExchange]
    current_topic: Optional[str]
    pending_actions: List[Dict]
    created_at: datetime
    last_updated: datetime

class MemoryManager:
    """Comprehensive memory management system"""
    
    def __init__(self, vector_store: VectorMemoryStore):
        self.vector_store = vector_store
        self.retrieval = EnhancedRetrieval(vector_store)
        self.active_sessions: Dict[str, SessionMemory] = {}
        self.memory_config = {
            "max_conversation_history": 50,
            "session_timeout_hours": 24,
            "auto_save_interval": 10,  # exchanges
            "context_memory_threshold": 0.7,
            "summarization_threshold": 20,  # exchanges before summarization
            "max_summary_length": 200,  # characters
            "preserve_recent_exchanges": 5  # keep most recent exchanges unsummarized
        }
        
        # Initialize with some basic knowledge
        self._initialize_base_knowledge()
        
        # Semantic clustering configuration
        self.clustering_config = {
            "similarity_threshold": 0.7,  # Minimum similarity to group memories
            "min_cluster_size": 2,        # Minimum memories per cluster
            "max_cluster_size": 10,       # Maximum memories per cluster
            "cluster_update_interval": 50 # Re-cluster after N new memories
        }
        self.memory_clusters: Dict[str, List[str]] = {}  # cluster_id -> [memory_ids]
        self.cluster_centroids: Dict[str, np.ndarray] = {}  # cluster_id -> centroid_vector
        self.memory_to_cluster: Dict[str, str] = {}  # memory_id -> cluster_id
        self.last_clustering_count = 0
    
    def _initialize_base_knowledge(self):
        """Initialize with basic Tuthand knowledge"""
        base_knowledge = [
            {
                "content": "Tuthand is an intelligent AI assistant for websites that provides helpful, accurate responses to visitor questions.",
                "metadata": {"source": "official_docs", "topic": "overview", "user_type": "all"},
                "confidence": 1.0
            },
            {
                "content": "Tuthand uses a trust-based routing system with three levels: auto_run (high confidence), confirm (medium confidence), and escalate (low confidence or sensitive topics).",
                "metadata": {"source": "official_docs", "topic": "trust_model", "user_type": "all"},
                "confidence": 1.0
            },
            {
                "content": "For developers: Tuthand provides technical implementation details and API documentation for integration.",
                "metadata": {"source": "official_docs", "topic": "development", "user_type": "developer"},
                "confidence": 0.9
            },
            {
                "content": "For customers: Tuthand offers user-friendly explanations and support for common questions.",
                "metadata": {"source": "official_docs", "topic": "support", "user_type": "customer"},
                "confidence": 0.9
            }
        ]
        
        for item in base_knowledge:
            self.vector_store.store_memory(
                content=item["content"],
                metadata=item["metadata"],
                source=item["metadata"]["source"],
                confidence=item["confidence"]
            )
        
        logger.info("Initialized base knowledge in vector store")
    
    def create_session(self, user_type: str = "customer") -> str:
        """Create a new user session"""
        session_id = str(uuid.uuid4())
        
        user_context = UserContext(
            user_type=user_type,
            session_id=session_id,
            confidence_history=[],
            topic_preferences=[],
            trust_level_history=[],
            last_interaction=datetime.now(),
            total_interactions=0
        )
        
        session_memory = SessionMemory(
            session_id=session_id,
            user_context=user_context,
            conversation_history=[],
            current_topic=None,
            pending_actions=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.active_sessions[session_id] = session_memory
        logger.info(f"Created new session {session_id} for user type {user_type}")
        return session_id
    
    def add_conversation_exchange(self, 
                                session_id: str,
                                query: str,
                                response: str,
                                user_type: str,
                                trust_level: str,
                                confidence: float,
                                strategy_used: str):
        """Add a conversation exchange to session memory"""
        if session_id not in self.active_sessions:
            # Create session if it doesn't exist
            self.create_session(user_type)
            session_id = list(self.active_sessions.keys())[-1]  # Get the newly created session
        
        session = self.active_sessions[session_id]
        
        # Create exchange
        exchange = ConversationExchange(
            query=query,
            response=response,
            user_type=user_type,
            trust_level=trust_level,
            confidence=confidence,
            timestamp=datetime.now(),
            strategy_used=strategy_used
        )
        
        # Add to conversation history
        session.conversation_history.append(exchange)
        
        # Update user context
        session.user_context.confidence_history.append(confidence)
        session.user_context.trust_level_history.append(trust_level)
        session.user_context.last_interaction = datetime.now()
        session.user_context.total_interactions += 1
        
        # Update current topic
        session.current_topic = self._extract_topic(query)
        
        # Update session
        session.last_updated = datetime.now()
        
        # Store in long-term memory if significant
        if confidence >= self.memory_config["context_memory_threshold"]:
            # Check for information conflicts before storing
            conflicts = self._detect_information_conflicts(query, session.user_context)
            if conflicts:
                self._resolve_information_conflicts(conflicts, query, confidence)
            
            self._store_interaction_memory(exchange, session.user_context)
        
        # Trim conversation history if too long
        max_history = self.memory_config["max_conversation_history"]
        if len(session.conversation_history) > max_history:
            session.conversation_history = session.conversation_history[-max_history:]
        
        logger.info(f"Added exchange to session {session_id}, confidence: {confidence}")
        
        # Check if conversation history needs summarization
        if len(session.conversation_history) >= self.memory_config["summarization_threshold"]:
            self._summarize_conversation_history(session_id)
        
        # Auto-save periodically
        if session.user_context.total_interactions % self.memory_config["auto_save_interval"] == 0:
            self._save_session_snapshot(session_id)
        
        # Check if we need to update semantic clusters
        self._check_and_update_clusters()
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        topics = {
            "pricing": ["price", "cost", "plan", "payment", "billing"],
            "features": ["feature", "capability", "function", "what can", "how does"],
            "integration": ["api", "integrate", "implement", "setup", "install"],
            "support": ["help", "problem", "issue", "error", "bug"],
            "documentation": ["docs", "guide", "tutorial", "documentation"]
        }
        
        query_lower = query.lower()
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                return topic
        
        return "general"
    
    def _store_interaction_memory(self, exchange: ConversationExchange, user_context: UserContext):
        """Store significant interactions in long-term memory"""
        # Store both query and response with different metadata
        query_metadata = {
            "interaction_type": "user_query",
            "user_type": user_context.user_type,
            "topic": self._extract_topic(exchange.query),
            "trust_level": exchange.trust_level,
            "strategy": exchange.strategy_used,
            "source": "user_interaction"
        }
        
        response_metadata = {
            "interaction_type": "assistant_response",
            "user_type": user_context.user_type,
            "topic": self._extract_topic(exchange.query),
            "trust_level": exchange.trust_level,
            "strategy": exchange.strategy_used,
            "source": "assistant_knowledge"
        }
        
        # Store query for learning user patterns
        self.vector_store.store_memory(
            content=exchange.query,
            metadata=query_metadata,
            source="user_interaction",
            confidence=exchange.confidence
        )
        
        # Store response for knowledge building
        self.vector_store.store_memory(
            content=exchange.response,
            metadata=response_metadata,
            source="assistant_knowledge",
            confidence=exchange.confidence
        )
    
    def get_session_context(self, session_id: str) -> Optional[Dict]:
        """Get current session context for enhanced retrieval"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Calculate average confidence
        confidence_history = session.user_context.confidence_history
        avg_confidence = sum(confidence_history) / len(confidence_history) if confidence_history else 0.5
        
        # Get recent topics
        recent_topics = []
        for exchange in session.conversation_history[-5:]:
            topic = self._extract_topic(exchange.query)
            if topic not in recent_topics:
                recent_topics.append(topic)
        
        return {
            "user_type": session.user_context.user_type,
            "avg_confidence": avg_confidence,
            "recent_topics": recent_topics,
            "current_topic": session.current_topic,
            "total_interactions": session.user_context.total_interactions,
            "session_duration": (datetime.now() - session.created_at).total_seconds() / 3600
        }
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for context"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        history = session.conversation_history[-limit:]
        
        return [
            {
                "query": exchange.query,
                "response": exchange.response,
                "confidence": exchange.confidence,
                "trust_level": exchange.trust_level,
                "timestamp": exchange.timestamp.isoformat()
            }
            for exchange in history
        ]
    
    def retrieve_relevant_context(self, 
                                session_id: str, 
                                query: str, 
                                top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context using enhanced retrieval"""
        # Get session context
        session_context = self.get_session_context(session_id)
        conversation_history = self.get_conversation_history(session_id)
        
        # Use enhanced retrieval
        results = self.retrieval.retrieve_context(
            query=query,
            user_context=session_context,
            conversation_history=conversation_history,
            top_k=top_k
        )
        
        return [
            {
                "content": result.content,
                "source": result.source,
                "confidence": result.confidence,
                "relevance_score": result.relevance_score,
                "retrieval_reason": result.retrieval_reason
            }
            for result in results
        ]
    
    def _summarize_conversation_history(self, session_id: str):
        """Summarize conversation history to reduce memory usage"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Don't summarize if conversation is too short
        if len(session.conversation_history) < self.memory_config["summarization_threshold"]:
            return
        
        # Preserve the most recent exchanges
        preserve_count = self.memory_config["preserve_recent_exchanges"]
        recent_exchanges = session.conversation_history[-preserve_count:]
        older_exchanges = session.conversation_history[:-preserve_count]
        
        if not older_exchanges:
            return  # Nothing to summarize
        
        # Create comprehensive summary
        summary = self._create_conversation_summary(older_exchanges, session.user_context)
        
        # Store summary in vector memory
        self.vector_store.store_memory(
            content=summary,
            metadata={
                "source": "conversation_summary",
                "session_id": session_id,
                "user_type": session.user_context.user_type,
                "exchanges_summarized": len(older_exchanges),
                "time_range": f"{older_exchanges[0].timestamp.isoformat()} to {older_exchanges[-1].timestamp.isoformat()}",
                "summary_type": "conversation_history"
            },
            source="memory_summarization",
            confidence=0.8
        )
        
        # Replace conversation history with recent exchanges only
        session.conversation_history = recent_exchanges
        
        logger.info(f"Summarized {len(older_exchanges)} exchanges for session {session_id}, keeping {len(recent_exchanges)} recent")
    
    def _detect_information_conflicts(self, new_content: str, user_context: UserContext) -> List[Dict]:
        """Detect conflicting information in memory"""
        conflicts = []
        
        # Check for personal information conflicts
        personal_patterns = [
            (r"my name is (\w+)", "name"),
            (r"i(?:'m| am) (\w+)", "name"), 
            (r"call me (\w+)", "name"),
            (r"i work at ([^.]+)", "company"),
            (r"i'm a ([^.]+)", "role"),
            (r"i live in ([^.]+)", "location"),
            (r"my email is ([^.]+)", "email")
        ]
        
        new_content_lower = new_content.lower()
        
        for pattern, info_type in personal_patterns:
            new_matches = re.findall(pattern, new_content_lower)
            if new_matches:
                new_value = new_matches[0].strip()
                
                # Search existing memories for same information type
                existing_memories = self.vector_store.search_memories(
                    query=f"{info_type} is",
                    top_k=10,
                    min_confidence=0.3
                )
                
                for memory in existing_memories:
                    existing_matches = re.findall(pattern, memory["content"].lower())
                    if existing_matches:
                        existing_value = existing_matches[0].strip()
                        if existing_value != new_value:
                            conflicts.append({
                                "type": info_type,
                                "new_value": new_value,
                                "existing_value": existing_value,
                                "existing_memory_id": memory["id"],
                                "conflict_severity": "high" if info_type in ["name", "email"] else "medium"
                            })
        
        return conflicts
    
    def _resolve_information_conflicts(self, conflicts: List[Dict], new_content: str, confidence: float):
        """Resolve information conflicts using correction signals"""
        
        for conflict in conflicts:
            # Check for explicit correction signals in new content
            correction_signals = [
                "you're mistaken", "that's wrong", "actually", "correction", 
                "i meant", "let me correct", "no, my", "sorry, my"
            ]
            
            new_content_lower = new_content.lower()
            is_explicit_correction = any(signal in new_content_lower for signal in correction_signals)
            
            # For personal information (name, email), treat new statements as corrections by default
            # since people typically don't change their name multiple times in a conversation
            is_personal_info_correction = (
                conflict["type"] in ["name", "email"] and 
                confidence > 0.8 and
                any(phrase in new_content_lower for phrase in ["my name is", "i'm", "i am", "call me"])
            )
            
            if is_explicit_correction or is_personal_info_correction:
                # Mark conflicting memory as outdated
                self._mark_memory_outdated(conflict["existing_memory_id"], 
                                         f"Corrected by user: {conflict['new_value']} (was {conflict['existing_value']})")
                
                logger.info(f"Resolved {conflict['type']} conflict: {conflict['existing_value']} → {conflict['new_value']}")
            
            elif confidence > 0.8:
                # High confidence new information - create temporal update
                self._create_temporal_update(conflict, new_content)
                logger.info(f"Created temporal update for {conflict['type']}: {conflict['new_value']}")
    
    def _mark_memory_outdated(self, memory_id: str, reason: str):
        """Mark a memory as outdated/corrected"""
        # Find and update the memory item
        for item in self.vector_store.memory_items:
            if item.id == memory_id:
                item.metadata["status"] = "outdated"
                item.metadata["outdated_reason"] = reason
                item.metadata["outdated_timestamp"] = datetime.now().isoformat()
                item.confidence = 0.1  # Lower confidence for outdated info
                break
        
        # Store a correction notice
        self.vector_store.store_memory(
            content=f"CORRECTION: {reason}",
            metadata={
                "source": "correction_notice",
                "corrected_memory_id": memory_id,
                "correction_type": "user_correction"
            },
            source="memory_correction",
            confidence=0.9
        )
    
    def _create_temporal_update(self, conflict: Dict, new_content: str):
        """Create temporal update for information that might change over time"""
        self.vector_store.store_memory(
            content=f"UPDATE: User's {conflict['type']} changed from {conflict['existing_value']} to {conflict['new_value']}. Latest: {new_content}",
            metadata={
                "source": "temporal_update",
                "information_type": conflict["type"],
                "previous_value": conflict["existing_value"],
                "new_value": conflict["new_value"],
                "update_type": "temporal_change"
            },
            source="memory_update",
            confidence=0.9
        )
    
    def _create_conversation_summary(self, exchanges: List[ConversationExchange], user_context: UserContext) -> str:
        """Create a comprehensive but concise summary of conversation exchanges"""
        
        # Extract key information patterns
        personal_info = []
        business_requirements = []
        technical_questions = []
        topics_discussed = []
        
        for exchange in exchanges:
            query_lower = exchange.query.lower()
            response_lower = exchange.response.lower()
            
            # Extract personal information
            personal_patterns = [
                (r"my name is (\w+)", "name"),
                (r"i(?:'m| am) (\w+)", "name"),
                (r"call me (\w+)", "name"),
                (r"i work at ([^.]+)", "company"),
                (r"i'm a ([^.]+)", "role"),
                (r"i live in ([^.]+)", "location"),
                (r"my email is ([^.]+)", "email"),
                (r"we have (\d+) (?:customers|users|visitors|employees)", "scale")
            ]
            
            for pattern, info_type in personal_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    personal_info.append(f"{info_type}: {match.strip()}")
            
            # Extract business requirements
            business_patterns = [
                r"we need ([^.]+)",
                r"looking for ([^.]+)",
                r"we want ([^.]+)",
                r"our goal is ([^.]+)",
                r"we're building ([^.]+)"
            ]
            
            for pattern in business_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    business_requirements.append(match.strip())
            
            # Extract technical questions
            tech_patterns = [
                r"how do(?:es)? (\w+) work",
                r"what(?:'s| is) (\w+)",
                r"can (?:you|it) (\w+)",
                r"api for (\w+)",
                r"integrate with (\w+)"
            ]
            
            for pattern in tech_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    technical_questions.append(match.strip())
            
            # Track topics
            topic = self._extract_topic(exchange.query)
            if topic not in topics_discussed:
                topics_discussed.append(topic)
        
        # Build summary
        summary_parts = []
        
        # User information
        if personal_info:
            unique_personal = list(set(personal_info))
            summary_parts.append(f"User details: {', '.join(unique_personal[:3])}")
        
        # Business requirements
        if business_requirements:
            unique_business = list(set(business_requirements))
            summary_parts.append(f"Requirements: {', '.join(unique_business[:2])}")
        
        # Technical interests
        if technical_questions:
            unique_tech = list(set(technical_questions))
            summary_parts.append(f"Technical interests: {', '.join(unique_tech[:2])}")
        
        # Topics covered
        if topics_discussed:
            summary_parts.append(f"Topics: {', '.join(topics_discussed)}")
        
        # Session statistics
        total_exchanges = len(exchanges)
        avg_confidence = sum(ex.confidence for ex in exchanges) / len(exchanges)
        time_span = (exchanges[-1].timestamp - exchanges[0].timestamp).total_seconds() / 60  # minutes
        
        summary_parts.append(f"Session: {total_exchanges} exchanges, {avg_confidence:.2f} avg confidence, {time_span:.0f}min duration")
        
        # Combine and truncate if needed
        full_summary = ". ".join(summary_parts)
        max_length = self.memory_config["max_summary_length"]
        
        if len(full_summary) > max_length:
            # Prioritize personal info and requirements
            essential_parts = []
            if personal_info:
                essential_parts.append(summary_parts[0])  # User details
            if business_requirements:
                for part in summary_parts:
                    if "Requirements:" in part:
                        essential_parts.append(part)
                        break
            
            essential_summary = ". ".join(essential_parts)
            if len(essential_summary) <= max_length:
                return essential_summary
            else:
                return essential_summary[:max_length-3] + "..."
        
        return full_summary
    
    def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get a summary of the current conversation"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        if not session.conversation_history:
            return "No conversation history available."
        
        # Create summary of current conversation
        return self._create_conversation_summary(session.conversation_history, session.user_context)
    
    def _save_session_snapshot(self, session_id: str):
        """Save session snapshot to persistent storage"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        snapshot = {
            "session_id": session_id,
            "snapshot_time": datetime.now().isoformat(),
            "user_context": asdict(session.user_context),
            "conversation_summary": {
                "total_exchanges": len(session.conversation_history),
                "avg_confidence": sum(ex.confidence for ex in session.conversation_history) / len(session.conversation_history) if session.conversation_history else 0,
                "main_topics": list(set(self._extract_topic(ex.query) for ex in session.conversation_history)),
                "trust_levels": list(set(ex.trust_level for ex in session.conversation_history))
            }
        }
        
        # Store snapshot in vector memory for future reference
        self.vector_store.store_memory(
            content=f"Session summary for user {session.user_context.user_type}: {snapshot['conversation_summary']}",
            metadata={
                "source": "session_snapshot",
                "session_id": session_id,
                "user_type": session.user_context.user_type,
                "snapshot_data": snapshot
            },
            source="session_management",
            confidence=0.8
        )
        
        logger.info(f"Saved session snapshot for {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        timeout = timedelta(hours=self.memory_config["session_timeout_hours"])
        cutoff_time = datetime.now() - timeout
        
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.last_updated < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            # Save final snapshot before cleanup
            self._save_session_snapshot(session_id)
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        active_sessions_count = len(self.active_sessions)
        total_interactions = sum(
            session.user_context.total_interactions 
            for session in self.active_sessions.values()
        )
        
        clustering_stats = {
            "total_clusters": len(self.memory_clusters),
            "clustered_memories": len(self.memory_to_cluster),
            "unclustered_memories": len(self.vector_store.memory_items) - len(self.memory_to_cluster),
            "avg_cluster_size": sum(len(cluster) for cluster in self.memory_clusters.values()) / len(self.memory_clusters) if self.memory_clusters else 0
        }
        
        return {
            "active_sessions": active_sessions_count,
            "total_interactions": total_interactions,
            "vector_store_stats": self.vector_store.get_memory_stats(),
            "retrieval_stats": self.retrieval.get_retrieval_stats(),
            "memory_config": self.memory_config,
            "clustering_stats": clustering_stats
        }
    
    def _check_and_update_clusters(self):
        """Check if clustering needs to be updated and perform clustering if needed"""
        current_memory_count = len(self.vector_store.memory_items)
        
        # Check if we should update clusters
        memories_added = current_memory_count - self.last_clustering_count
        if memories_added >= self.clustering_config["cluster_update_interval"]:
            logger.info(f"Updating semantic clusters: {memories_added} new memories since last clustering")
            self._perform_semantic_clustering()
            self.last_clustering_count = current_memory_count
    
    def _perform_semantic_clustering(self):
        """Perform semantic clustering on all memories"""
        try:
            # Get all memory items with embeddings
            memories_with_embeddings = []
            memory_ids = []
            
            for memory_item in self.vector_store.memory_items:
                if memory_item.embedding:
                    memories_with_embeddings.append(np.array(memory_item.embedding))
                    memory_ids.append(memory_item.id)
            
            if len(memories_with_embeddings) < self.clustering_config["min_cluster_size"]:
                logger.info(f"Not enough memories ({len(memories_with_embeddings)}) for clustering")
                return
            
            # Perform clustering using cosine similarity
            similarity_matrix = self._compute_similarity_matrix(memories_with_embeddings)
            clusters = self._agglomerative_clustering(similarity_matrix, memory_ids)
            
            # Update cluster data structures
            self._update_cluster_mappings(clusters, memories_with_embeddings, memory_ids)
            
            logger.info(f"Created {len(clusters)} semantic clusters from {len(memory_ids)} memories")
            
        except Exception as e:
            logger.error(f"Error during semantic clustering: {e}")
    
    def _compute_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarity matrix for embeddings"""
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Compute cosine similarity
                    vec1, vec2 = embeddings[i], embeddings[j]
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _agglomerative_clustering(self, similarity_matrix: np.ndarray, memory_ids: List[str]) -> List[List[str]]:
        """Perform agglomerative clustering based on similarity threshold"""
        n = len(memory_ids)
        clusters = [[memory_ids[i]] for i in range(n)]  # Start with each memory in its own cluster
        
        threshold = self.clustering_config["similarity_threshold"]
        max_cluster_size = self.clustering_config["max_cluster_size"]
        
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            best_similarity = -1
            best_pair = None
            
            # Find the most similar pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if len(clusters[i]) + len(clusters[j]) > max_cluster_size:
                        continue  # Skip if merged cluster would be too large
                    
                    # Compute average similarity between clusters
                    similarities = []
                    for mem_i_id in clusters[i]:
                        for mem_j_id in clusters[j]:
                            idx_i = memory_ids.index(mem_i_id)
                            idx_j = memory_ids.index(mem_j_id)
                            similarities.append(similarity_matrix[idx_i][idx_j])
                    
                    avg_similarity = np.mean(similarities)
                    
                    if avg_similarity > threshold and avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_pair = (i, j)
            
            # Merge the best pair if found
            if best_pair:
                i, j = best_pair
                clusters[i].extend(clusters[j])
                clusters.pop(j)
                merged = True
        
        # Filter out clusters that are too small
        min_cluster_size = self.clustering_config["min_cluster_size"]
        filtered_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
        
        return filtered_clusters
    
    def _update_cluster_mappings(self, clusters: List[List[str]], embeddings: List[np.ndarray], memory_ids: List[str]):
        """Update cluster data structures with new clustering results"""
        # Clear existing mappings
        self.memory_clusters.clear()
        self.cluster_centroids.clear()
        self.memory_to_cluster.clear()
        
        for i, cluster in enumerate(clusters):
            cluster_id = f"cluster_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store cluster
            self.memory_clusters[cluster_id] = cluster
            
            # Update memory-to-cluster mapping
            for memory_id in cluster:
                self.memory_to_cluster[memory_id] = cluster_id
            
            # Compute cluster centroid
            cluster_embeddings = []
            for memory_id in cluster:
                memory_idx = memory_ids.index(memory_id)
                cluster_embeddings.append(embeddings[memory_idx])
            
            if cluster_embeddings:
                centroid = np.mean(cluster_embeddings, axis=0)
                self.cluster_centroids[cluster_id] = centroid
    
    def get_cluster_context(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant clusters for a query based on embedding similarity"""
        if not self.cluster_centroids:
            return []
        
        cluster_similarities = []
        
        for cluster_id, centroid in self.cluster_centroids.items():
            # Compute similarity between query and cluster centroid
            similarity = np.dot(query_embedding, centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(centroid)
            )
            
            cluster_similarities.append({
                "cluster_id": cluster_id,
                "similarity": similarity,
                "memory_ids": self.memory_clusters[cluster_id],
                "cluster_size": len(self.memory_clusters[cluster_id])
            })
        
        # Sort by similarity and return top_k
        cluster_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return cluster_similarities[:top_k]
    
    def get_cluster_summary(self, cluster_id: str) -> Optional[str]:
        """Generate a summary of memories in a cluster"""
        if cluster_id not in self.memory_clusters:
            return None
        
        memory_ids = self.memory_clusters[cluster_id]
        contents = []
        topics = []
        sources = []
        
        for memory_item in self.vector_store.memory_items:
            if memory_item.id in memory_ids:
                contents.append(memory_item.content)
                if hasattr(memory_item.metadata, 'topic'):
                    topics.append(memory_item.metadata.get('topic', 'unknown'))
                sources.append(memory_item.source)
        
        # Analyze cluster characteristics
        unique_topics = list(set(topics))
        unique_sources = list(set(sources))
        
        # Create summary
        summary_parts = []
        summary_parts.append(f"Cluster with {len(memory_ids)} related memories")
        
        if unique_topics:
            summary_parts.append(f"Topics: {', '.join(unique_topics[:3])}")
        
        if unique_sources:
            summary_parts.append(f"Sources: {', '.join(unique_sources[:3])}")
        
        # Add sample content
        if contents:
            sample_content = contents[0][:100] + "..." if len(contents[0]) > 100 else contents[0]
            summary_parts.append(f"Example: {sample_content}")
        
        return ". ".join(summary_parts)

# Factory function
def create_memory_manager(vector_store: VectorMemoryStore) -> MemoryManager:
    """Create a memory manager with vector store"""
    return MemoryManager(vector_store)