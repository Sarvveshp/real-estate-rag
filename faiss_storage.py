import faiss
import numpy as np
from typing import List, Dict, Tuple
import json

class FAISSStorage:
    def __init__(self, dimension: int = 1536):  # OpenAI embedding dimension
        """Initialize FAISS index and storage."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
    def add_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """
        Add embeddings and their metadata to FAISS.
        
        Args:
            embeddings: List of numpy arrays containing embeddings
            metadata: List of dictionaries containing metadata for each embedding
        """
        embeddings_array = np.vstack(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.metadata.extend(metadata)
        
        # Save metadata to file
        with open('faiss_metadata.json', 'w') as f:
            json.dump(self.metadata, f)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """
        Search for k nearest neighbors to the query embedding.
        
        Args:
            query_embedding: Query embedding as numpy array
            k: Number of results to return
            
        Returns:
            List of tuples containing (distance, metadata) for each result
        """
        query_array = np.expand_dims(query_embedding, axis=0).astype('float32')
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((dist, self.metadata[idx]))
        
        return results
    
    def save(self, path: str):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)
        
    def load(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        
        # Load metadata
        try:
            with open('faiss_metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = []
