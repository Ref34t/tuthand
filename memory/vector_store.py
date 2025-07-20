#!/usr/bin/env python3
"""
Vector Memory Store - Week 3 Implementation
Supporting Pinecone, Chroma, and in-memory vector databases
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Optional imports for different vector stores
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """A single memory item with metadata"""
    id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
    confidence: float = 1.0

class VectorMemoryStore:
    """Vector-based memory storage with multiple backend support"""
    
    def __init__(self, 
                 provider: str = "inmemory",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_name: str = "tuthand-memory"):
        self.provider = provider.lower()
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.memory_items: List[MemoryItem] = []
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.encoder = None
            self.embedding_dim = 384
        
        # Initialize vector store backend
        self.vector_store = self._init_vector_store()
        
    def _init_vector_store(self):
        """Initialize the appropriate vector store backend"""
        if self.provider == "pinecone":
            return self._init_pinecone()
        elif self.provider == "chroma":
            return self._init_chroma()
        else:
            return self._init_inmemory()
    
    def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        if not PINECONE_AVAILABLE:
            logger.error("Pinecone not available, falling back to in-memory")
            return self._init_inmemory()
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.error("PINECONE_API_KEY not set, falling back to in-memory")
            return self._init_inmemory()
        
        try:
            pinecone.init(api_key=api_key)
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
            
            return pinecone.Index(self.index_name)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return self._init_inmemory()
    
    def _init_chroma(self):
        """Initialize ChromaDB vector store"""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available, falling back to in-memory")
            return self._init_inmemory()
        
        try:
            client = chromadb.Client()
            collection = client.get_or_create_collection(
                name=self.index_name,
                metadata={"hnsw:space": "cosine"}
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return self._init_inmemory()
    
    def _init_inmemory(self):
        """Initialize in-memory vector store (fallback)"""
        logger.info("Using in-memory vector store")
        return {"type": "inmemory", "vectors": {}, "metadata": {}}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.encoder:
            embedding = self.encoder.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        else:
            # Mock embedding for testing
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            # Create a deterministic "embedding" from hash
            np.random.seed(int(hash_obj.hexdigest()[:8], 16))
            return np.random.rand(self.embedding_dim).tolist()
    
    def store_memory(self, 
                    content: str, 
                    metadata: Dict[str, Any],
                    source: str = "user_interaction",
                    confidence: float = 1.0) -> str:
        """Store a memory item in the vector store"""
        memory_id = str(uuid.uuid4())
        embedding = self.generate_embedding(content)
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            timestamp=datetime.now(),
            source=source,
            confidence=confidence
        )
        
        # Store in backend
        if self.provider == "pinecone":
            self._store_pinecone(memory_item)
        elif self.provider == "chroma":
            self._store_chroma(memory_item)
        else:
            self._store_inmemory(memory_item)
        
        # Keep local copy for quick access
        self.memory_items.append(memory_item)
        
        logger.info(f"Stored memory item {memory_id} from source {source}")
        return memory_id
    
    def _store_pinecone(self, item: MemoryItem):
        """Store in Pinecone"""
        try:
            self.vector_store.upsert([
                (item.id, item.embedding, {
                    "content": item.content,
                    "source": item.source,
                    "timestamp": item.timestamp.isoformat(),
                    "confidence": item.confidence,
                    **item.metadata
                })
            ])
        except Exception as e:
            logger.error(f"Failed to store in Pinecone: {e}")
    
    def _store_chroma(self, item: MemoryItem):
        """Store in ChromaDB"""
        try:
            self.vector_store.add(
                ids=[item.id],
                embeddings=[item.embedding],
                documents=[item.content],
                metadatas=[{
                    "source": item.source,
                    "timestamp": item.timestamp.isoformat(),
                    "confidence": item.confidence,
                    **item.metadata
                }]
            )
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
    
    def _store_inmemory(self, item: MemoryItem):
        """Store in memory"""
        self.vector_store["vectors"][item.id] = item.embedding
        self.vector_store["metadata"][item.id] = {
            "content": item.content,
            "source": item.source,
            "timestamp": item.timestamp.isoformat(),
            "confidence": item.confidence,
            **item.metadata
        }
    
    def search_memories(self, 
                       query: str, 
                       top_k: int = 5,
                       filter_metadata: Optional[Dict] = None,
                       min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Search for relevant memories"""
        query_embedding = self.generate_embedding(query)
        
        if self.provider == "pinecone":
            return self._search_pinecone(query_embedding, top_k, filter_metadata, min_confidence)
        elif self.provider == "chroma":
            return self._search_chroma(query_embedding, top_k, filter_metadata, min_confidence)
        else:
            return self._search_inmemory(query_embedding, top_k, filter_metadata, min_confidence)
    
    def _search_pinecone(self, query_embedding, top_k, filter_metadata, min_confidence):
        """Search Pinecone"""
        try:
            filter_dict = {}
            if filter_metadata:
                filter_dict.update(filter_metadata)
            if min_confidence > 0:
                filter_dict["confidence"] = {"$gte": min_confidence}
            
            results = self.vector_store.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            return [
                {
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def _search_chroma(self, query_embedding, top_k, filter_metadata, min_confidence):
        """Search ChromaDB"""
        try:
            where_clause = {}
            if filter_metadata:
                where_clause.update(filter_metadata)
            if min_confidence > 0:
                where_clause["confidence"] = {"$gte": min_confidence}
            
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                search_results.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "score": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata
                })
            
            return search_results
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _search_inmemory(self, query_embedding, top_k, filter_metadata, min_confidence):
        """Search in-memory store"""
        try:
            results = []
            
            for item_id, embedding in self.vector_store["vectors"].items():
                metadata = self.vector_store["metadata"][item_id]
                
                # Apply filters
                if min_confidence > 0 and metadata.get("confidence", 0) < min_confidence:
                    continue
                
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                results.append({
                    "id": item_id,
                    "content": metadata.get("content", ""),
                    "score": similarity,
                    "metadata": metadata
                })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"In-memory search failed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            "provider": self.provider,
            "total_memories": len(self.memory_items),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dim,
            "last_update": datetime.now().isoformat()
        }
    
    def clear_memories(self, filter_metadata: Optional[Dict] = None):
        """Clear memories (use with caution)"""
        if filter_metadata:
            # Remove specific memories
            self.memory_items = [
                item for item in self.memory_items
                if not all(item.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        else:
            # Clear all memories
            self.memory_items.clear()
            
        logger.warning(f"Cleared memories with filter: {filter_metadata}")

# Factory function for easy instantiation
def create_vector_store(provider: str = "inmemory", **kwargs) -> VectorMemoryStore:
    """Create a vector memory store with the specified provider"""
    return VectorMemoryStore(provider=provider, **kwargs)