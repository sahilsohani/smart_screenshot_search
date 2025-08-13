"""
Searcher for Smart Screenshot Search.

This module handles hybrid search combining text and image embeddings
with tunable weighting between text and visual similarity.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports with fallbacks
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

# Configure logging
logger = logging.getLogger(__name__)


class IndexNotFoundError(Exception):
    """Raised when required index files are not found."""
    pass


class InvalidIndexError(Exception):
    """Raised when index files are corrupted or invalid."""
    pass


class HybridSearcher:
    """
    Hybrid searcher combining text and image embeddings for screenshot search.
    
    Supports both FAISS indexes for fast search and NumPy fallback for
    environments without FAISS. Provides tunable weighting between text
    and image similarity scores.
    """
    
    def __init__(self, index_dir: Union[str, Path]):
        """
        Initialize the hybrid searcher.
        
        Args:
            index_dir: Directory containing the index files
            
        Raises:
            IndexNotFoundError: If index directory or metadata not found
            InvalidIndexError: If index files are corrupted
        """
        self.index_dir = Path(index_dir)
        
        # Validate index directory
        if not self.index_dir.exists():
            raise IndexNotFoundError(f"Index directory not found: {self.index_dir}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load embeddings and indexes
        self._load_indexes()
        
        # Load models for query encoding
        self._load_models()
        
        logger.info(f"HybridSearcher initialized with {self.metadata['num_images']} images")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load index metadata from meta.json.
        
        Returns:
            Metadata dictionary
            
        Raises:
            IndexNotFoundError: If meta.json not found
            InvalidIndexError: If meta.json is invalid
        """
        meta_path = self.index_dir / "meta.json"
        
        if not meta_path.exists():
            raise IndexNotFoundError(f"Index metadata not found: {meta_path}")
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate required fields
            required_fields = ['num_images', 'built_at', 'items']
            for field in required_fields:
                if field not in metadata:
                    raise InvalidIndexError(f"Missing required field in metadata: {field}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise InvalidIndexError(f"Invalid JSON in metadata: {e}")
        except Exception as e:
            raise InvalidIndexError(f"Failed to load metadata: {e}")
    
    def _load_indexes(self) -> None:
        """
        Load text and image indexes (FAISS or NumPy fallback).
        
        Raises:
            IndexNotFoundError: If no valid index files found
            InvalidIndexError: If index files are corrupted
        """
        # Try to load FAISS indexes first
        self.text_index = None
        self.image_index = None
        self.text_embeddings = None
        self.image_embeddings = None
        self.use_faiss = False
        
        if HAVE_FAISS and self.metadata.get('have_faiss', False):
            try:
                text_faiss_path = self.index_dir / "text_index.faiss"
                image_faiss_path = self.index_dir / "image_index.faiss"
                
                if text_faiss_path.exists() and image_faiss_path.exists():
                    self.text_index = faiss.read_index(str(text_faiss_path))
                    self.image_index = faiss.read_index(str(image_faiss_path))
                    self.use_faiss = True
                    logger.info("Loaded FAISS indexes")
                    return
            except Exception as e:
                logger.warning(f"Failed to load FAISS indexes: {e}")
        
        # Fallback to NumPy
        try:
            text_numpy_path = self.index_dir / "text_index.npy"
            image_numpy_path = self.index_dir / "image_index.npy"
            
            if not text_numpy_path.exists():
                raise IndexNotFoundError(f"Text index not found: {text_numpy_path}")
            if not image_numpy_path.exists():
                raise IndexNotFoundError(f"Image index not found: {image_numpy_path}")
            
            self.text_embeddings = np.load(text_numpy_path)
            self.image_embeddings = np.load(image_numpy_path)
            
            # Validate shapes
            expected_items = self.metadata['num_images']
            if len(self.text_embeddings) != expected_items:
                raise InvalidIndexError(f"Text embeddings count mismatch: {len(self.text_embeddings)} != {expected_items}")
            if len(self.image_embeddings) != expected_items:
                raise InvalidIndexError(f"Image embeddings count mismatch: {len(self.image_embeddings)} != {expected_items}")
            
            logger.info("Loaded NumPy embeddings")
            
        except Exception as e:
            raise InvalidIndexError(f"Failed to load embeddings: {e}")
    
    def _load_models(self) -> None:
        """Load text and image encoding models."""
        logger.info("Loading query encoding models...")
        
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.image_model = SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            raise RuntimeError(f"Failed to load encoding models: {e}")
    
    def _encode_text_query(self, query: str) -> np.ndarray:
        """
        Encode text query into embedding.
        
        Args:
            query: Text query string
            
        Returns:
            Normalized text embedding
        """
        if not query.strip():
            # Return zero vector for empty query
            dim = self.metadata.get('text_embedding_dim', 384)  # MiniLM default
            return np.zeros(dim, dtype=np.float32)
        
        embedding = self.text_model.encode([query], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
    
    def _encode_image_query(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Encode image query into embedding.
        
        Args:
            image_path: Path to query image
            
        Returns:
            Normalized image embedding
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                embedding = self.image_model.encode([img], normalize_embeddings=True)[0]
                return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode image query {image_path}: {e}")
            # Return zero vector on failure
            dim = self.metadata.get('image_embedding_dim', 512)  # CLIP default
            return np.zeros(dim, dtype=np.float32)
    
    def _search_text(self, text_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search text embeddings for similar items.
        
        Args:
            text_embedding: Query text embedding
            k: Number of results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if self.use_faiss:
            # FAISS search (inner product for normalized vectors)
            scores, indices = self.text_index.search(text_embedding.reshape(1, -1), k)
            return scores[0], indices[0]
        else:
            # NumPy search (manual inner product)
            scores = np.dot(self.text_embeddings, text_embedding)
            indices = np.argsort(scores)[::-1][:k]  # Top k indices
            return scores[indices], indices
    
    def _search_image(self, image_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search image embeddings for similar items.
        
        Args:
            image_embedding: Query image embedding
            k: Number of results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if self.use_faiss:
            # FAISS search (inner product for normalized vectors)
            scores, indices = self.image_index.search(image_embedding.reshape(1, -1), k)
            return scores[0], indices[0]
        else:
            # NumPy search (manual inner product)
            scores = np.dot(self.image_embeddings, image_embedding)
            indices = np.argsort(scores)[::-1][:k]  # Top k indices
            return scores[indices], indices
    
    def _normalize_scores_simple(self, scores: np.ndarray) -> np.ndarray:
        """
        Simple score normalization that preserves relative differences.
        
        Args:
            scores: Raw similarity scores
            
        Returns:
            Scores normalized to [0, 1] while preserving ranking
        """
        if len(scores) == 0:
            return scores
        
        # Ensure non-negative (cosine similarity should be 0-1 for normalized vectors)
        scores = np.maximum(scores, 0)
        
        # Simple min-max normalization
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        # Just normalize to 0-1 range, preserving relative differences
        normalized = (scores - min_score) / (max_score - min_score)
        
        return normalized
    
    def _create_snippet(self, ocr_text: str, max_length: int = 300) -> str:
        """
        Create a text snippet from OCR text.
        
        Args:
            ocr_text: Full OCR text
            max_length: Maximum snippet length
            
        Returns:
            Truncated text snippet
        """
        if not ocr_text:
            return ""
        
        # Clean up text
        text = ocr_text.strip()
        
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        snippet = text[:max_length]
        last_space = snippet.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can break at >80% of max length
            snippet = snippet[:last_space]
        
        return snippet + "..." if len(text) > max_length else snippet
    
    def search(self, query: str, top_k: int = 12, alpha: float = 0.6, 
               image_query: Optional[Union[str, Path]] = None,
               min_score_threshold: float = 0.08) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and image similarities with quality filtering.
        
        Args:
            query: Text query string
            top_k: Number of results to return
            alpha: Weight for text vs image scores (0.0 = image only, 1.0 = text only)
            image_query: Optional path to image for image-based search
            min_score_threshold: Minimum score threshold for results (default: 0.15)
            
        Returns:
            List of result dictionaries with keys: path, score, tags, snippet, filename
            
        Raises:
            ValueError: If alpha not in [0, 1] or top_k invalid
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        # Search a reasonable number - not too many to avoid noise
        search_k = min(max(top_k * 3, 50), self.metadata['num_images'])
        
        # Encode queries
        text_embedding = self._encode_text_query(query)
        
        if image_query:
            image_embedding = self._encode_image_query(image_query)
        else:
            # Use zero embedding if no image query
            dim = self.metadata.get('image_embedding_dim', 512)
            image_embedding = np.zeros(dim, dtype=np.float32)
        
        # Search both modalities
        text_scores, text_indices = self._search_text(text_embedding, search_k)
        image_scores, image_indices = self._search_image(image_embedding, search_k)
        
        # Use simple normalization to preserve score differences
        text_scores_norm = self._normalize_scores_simple(text_scores)
        image_scores_norm = self._normalize_scores_simple(image_scores)
        
        # Combine results from both searches
        combined_scores = {}
        
        # Add text results - use raw scores for filtering but normalized for final computation
        for score_norm, score_raw, idx in zip(text_scores_norm, text_scores, text_indices):
            if idx < len(self.metadata['items']) and score_raw >= 0.05:  # Filter on raw similarity
                combined_scores[idx] = {
                    'text_score': score_norm, 
                    'image_score': 0.0, 
                    'raw_text_score': score_raw
                }
        
        # Add image results - use raw scores for filtering but normalized for final computation
        for score_norm, score_raw, idx in zip(image_scores_norm, image_scores, image_indices):
            if idx < len(self.metadata['items']) and score_raw >= 0.05:  # Filter on raw similarity
                if idx in combined_scores:
                    combined_scores[idx]['image_score'] = score_norm
                    combined_scores[idx]['raw_image_score'] = score_raw
                else:
                    combined_scores[idx] = {
                        'text_score': 0.0, 
                        'image_score': score_norm, 
                        'raw_text_score': 0.0, 
                        'raw_image_score': score_raw
                    }
        
        # Compute final scores and create results
        results = []
        
        for idx, scores in combined_scores.items():
            # Hybrid score: alpha * text + (1-alpha) * image
            final_score = alpha * scores['text_score'] + (1 - alpha) * scores['image_score']
            
            # Apply threshold filter on final score
            if final_score < min_score_threshold:
                continue
            
            # Get item metadata
            item = self.metadata['items'][idx]
            
            # Additional quality check: if query has meaningful text, ensure some text relevance
            if query.strip() and alpha > 0.3:  # Only if we care about text significantly
                raw_text_score = scores.get('raw_text_score', 0)
                raw_image_score = scores.get('raw_image_score', 0)
                
                # For text queries, require either decent text match OR very strong image match
                if raw_text_score < 0.1 and raw_image_score < 0.4:
                    continue
            
            # Create result
            result = {
                'path': item['path'],
                'filename': item['filename'],
                'score': float(final_score),
                'text_score': float(scores['text_score']),
                'image_score': float(scores['image_score']),
                'tags': item.get('tags', []),
                'snippet': self._create_snippet(item.get('ocr_text', '')),
                'file_size': item.get('file_size', 0),
                'sha1': item.get('sha1', ''),
                'modified_time': item.get('modified_time', 0)
            }
            
            results.append(result)
        
        # Sort by final score (descending) and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Additional post-processing: boost results that match query semantically
        if query.strip() and results:
            query_words = set(query.lower().split())
            
            for result in results:
                # Check if query words appear in OCR text or tags
                text_content = (result.get('snippet', '') + ' ' + ' '.join(result.get('tags', []))).lower()
                word_matches = sum(1 for word in query_words if word in text_content)
                
                if word_matches > 0:
                    # Boost score slightly for direct word matches
                    boost = min(0.1, word_matches * 0.03)
                    result['score'] = min(1.0, result['score'] + boost)
            
            # Re-sort after boosting
            results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Search found {len(results)} relevant results for '{query}'")
        
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get searcher statistics and information.
        
        Returns:
            Dictionary with searcher stats
        """
        return {
            'num_images': self.metadata['num_images'],
            'built_at': self.metadata['built_at'],
            'use_faiss': self.use_faiss,
            'screenshots_dir': self.metadata.get('screenshots_dir', ''),
            'text_embedding_dim': self.metadata.get('text_embedding_dim', 0),
            'image_embedding_dim': self.metadata.get('image_embedding_dim', 0),
            'index_dir': str(self.index_dir)
        }


def create_searcher(index_dir: Union[str, Path]) -> HybridSearcher:
    """
    Convenience function to create a HybridSearcher.
    
    Args:
        index_dir: Directory containing index files
        
    Returns:
        Initialized HybridSearcher instance
    """
    return HybridSearcher(index_dir)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Smart Screenshot Search')
    parser.add_argument('--index', required=True, help='Index directory')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--alpha', type=float, default=0.6, help='Text/image weight (0.0-1.0)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--image', help='Optional image query path')
    parser.add_argument('--min-score', type=float, default=0.15, help='Minimum score threshold')
    
    args = parser.parse_args()
    
    try:
        # Create searcher
        searcher = HybridSearcher(args.index)
        
        print(f"=== Search Statistics ===")
        stats = searcher.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\n=== Search Results ===")
        print(f"Query: '{args.query}'")
        print(f"Alpha: {args.alpha} (text weight)")
        print(f"Min Score Threshold: {args.min_score}")
        
        # Perform search
        results = searcher.search(
            query=args.query, 
            top_k=args.top_k, 
            alpha=args.alpha,
            image_query=args.image,
            min_score_threshold=args.min_score
        )
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} (score: {result['score']:.3f})")
            print(f"   Text: {result['text_score']:.3f}, Image: {result['image_score']:.3f}")
            print(f"   Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
            if result['snippet']:
                print(f"   Text: {result['snippet'][:100]}...")
            print(f"   Path: {result['path']}")
        
        if not results:
            print("No results found above the minimum threshold.")
            
    except Exception as e:
        print(f"Search failed: {e}")
        exit(1)