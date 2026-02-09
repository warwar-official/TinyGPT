import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Tuple, Dict
import os


class SimpleVectorDB:
    """Simple vector database for RAG applications with Ukrainian language support."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        """
        Initialize the vector database.
        
        Args:
            model_name: HuggingFace model name. Default supports Ukrainian.
            device: 'cpu' or 'cuda'
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
        self.chunks = []  # Store actual text chunks
        self.metadata = []  # Store metadata for each chunk
        
    def add_chunks(self, chunks: List[str], metadata: List[Dict] = None):
        """
        Add text chunks to the database.
        
        Args:
            chunks: List of text strings to add
            metadata: Optional list of metadata dicts for each chunk
        """
        if not chunks:
            return
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunks))
    
    def remove_chunks(self, indices: List[int]):
        """
        Remove chunks by their indices.
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        
        Args:
            indices: List of chunk indices to remove
        """
        # Create set for O(1) lookup
        indices_set = set(indices)
        
        # Keep chunks that are not in removal list
        new_chunks = [chunk for i, chunk in enumerate(self.chunks) if i not in indices_set]
        new_metadata = [meta for i, meta in enumerate(self.metadata) if i not in indices_set]
        
        # Rebuild index
        self.chunks = []
        self.metadata = []
        self.index.reset()
        
        if new_chunks:
            self.add_chunks(new_chunks, new_metadata)
    
    def modify_chunk(self, index: int, new_text: str, new_metadata: Dict = None):
        """
        Modify a chunk at given index.
        
        Args:
            index: Index of chunk to modify
            new_text: New text for the chunk
            new_metadata: Optional new metadata
        """
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Index {index} out of range")
        
        # Update chunk and metadata
        self.chunks[index] = new_text
        if new_metadata is not None:
            self.metadata[index] = new_metadata
        
        # Rebuild index (FAISS doesn't support in-place updates)
        self._rebuild_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for most similar chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
        
        Returns:
            List of tuples: (chunk_text, distance, metadata)
        """
        if len(self.chunks) == 0:
            return []
        
        # Limit top_k to available chunks
        top_k = min(top_k, len(self.chunks))
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((
                self.chunks[idx],
                float(dist),
                self.metadata[idx]
            ))
        
        return results
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from current chunks."""
        temp_chunks = self.chunks.copy()
        temp_metadata = self.metadata.copy()
        
        self.chunks = []
        self.metadata = []
        self.index.reset()
        
        if temp_chunks:
            self.add_chunks(temp_chunks, temp_metadata)
    
    def save(self, path: str):
        """
        Save the vector database to disk.
        
        Args:
            path: Directory path to save the database
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save chunks and metadata
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
    
    def load(self, path: str):
        """
        Load the vector database from disk.
        
        Args:
            path: Directory path containing the saved database
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load chunks and metadata
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "model": self.model._model_config['_name_or_path']
        }
