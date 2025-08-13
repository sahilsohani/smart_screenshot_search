"""
Indexer for Smart Screenshot Search.

This module handles building search indexes from screenshot folders,
including OCR text extraction, embedding generation, and FAISS/NumPy storage.
"""

import argparse
import json
import logging
import os
import ssl
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Handle SSL certificate issues (temporary workaround)
if os.environ.get("OFFLINE") == "1" or os.environ.get("PYTHONHTTPSVERIFY") == "0":
    ssl._create_default_https_context = ssl._create_unverified_context

from .utils import list_images, file_sha1, simple_tags, compute_phash

# Optional imports with fallbacks
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    warnings.warn("FAISS not available, falling back to NumPy storage")

try:
    import easyocr
    HAVE_EASYOCR = True
except ImportError:
    HAVE_EASYOCR = False
    raise ImportError("EasyOCR is required for text extraction")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageIndexer:
    """
    Handles the indexing of screenshots with OCR text extraction and embedding generation.
    """
    
    def __init__(self):
        """Initialize the indexer with lazy model loading."""
        self.text_model: Optional[SentenceTransformer] = None
        self.image_model: Optional[SentenceTransformer] = None
        self.ocr_reader: Optional[Any] = None
        
    def _load_models(self) -> None:
        """Load all required models. Called once when first needed."""
        if self.text_model is None:
            logger.info("Loading text embedding model (all-MiniLM-L6-v2)...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        if self.image_model is None:
            logger.info("Loading image embedding model (clip-ViT-B-32)...")
            self.image_model = SentenceTransformer('clip-ViT-B-32')
            
        if self.ocr_reader is None:
            logger.info("Loading EasyOCR reader (English, CPU)...")
            # Handle SSL issues for EasyOCR downloads
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    logger.warning("SSL issue with EasyOCR, trying with unverified SSL context...")
                    # Temporarily disable SSL verification for EasyOCR
                    original_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                    try:
                        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                        logger.info("EasyOCR loaded successfully with SSL workaround")
                    finally:
                        # Restore original SSL context
                        ssl._create_default_https_context = original_context
                else:
                    raise e

    def _load_cache(self, cache_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load existing cache from disk.
        
        Args:
            cache_path: Path to cache.json file
            
        Returns:
            Dictionary mapping SHA-1 hashes to cached data
        """
        if not cache_path.exists():
            logger.info("No existing cache found, starting fresh")
            return {}
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Loaded cache with {len(cache)} entries from {cache_path}")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return {}

    def _save_cache(self, cache: Dict[str, Dict[str, Any]], cache_path: Path) -> None:
        """
        Save cache to disk.
        
        Args:
            cache: Cache dictionary to save
            cache_path: Path where to save cache.json
        """
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cache with {len(cache)} entries to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")

    def _extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        try:
            results = self.ocr_reader.readtext(str(image_path))
            # Extract text from results (each result is [bbox, text, confidence])
            text_parts = [result[1] for result in results if result[2] > 0.5]  # Confidence threshold
            return ' '.join(text_parts).strip()
        except Exception as e:
            logger.warning(f"OCR failed for {image_path.name}: {e}")
            return ""
    
    def _create_text_embedding(self, text: str) -> np.ndarray:
        """
        Create normalized text embedding.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        embedding = self.text_model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
    
    def _create_image_embedding(self, image_path: Path) -> np.ndarray:
        """
        Create normalized image embedding using CLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized embedding vector
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                embedding = self.image_model.encode([img], normalize_embeddings=True)[0]
                return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Image embedding failed for {image_path.name}: {e}")
            # Return zero vector on failure
            return np.zeros(self.image_model.get_sentence_embedding_dimension(), dtype=np.float32)

    def _batch_encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Batch encode multiple texts for better performance.
        
        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Encoding text batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Handle empty texts
            batch_texts_processed = []
            empty_indices = []
            for j, text in enumerate(batch_texts):
                if not text.strip():
                    empty_indices.append(j)
                    batch_texts_processed.append("empty")  # Placeholder
                else:
                    batch_texts_processed.append(text)
            
            # Encode batch
            try:
                batch_embeddings = self.text_model.encode(batch_texts_processed, normalize_embeddings=True)
                batch_embeddings = batch_embeddings.astype(np.float32)
                
                # Replace empty text embeddings with zero vectors
                for empty_idx in empty_indices:
                    batch_embeddings[empty_idx] = np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.warning(f"Batch text encoding failed: {e}")
                # Fallback to individual encoding
                for text in batch_texts:
                    try:
                        if not text.strip():
                            embedding = np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
                        else:
                            embedding = self.text_model.encode([text], normalize_embeddings=True)[0]
                            embedding = embedding.astype(np.float32)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Individual text encoding failed: {e2}")
                        all_embeddings.append(np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32))
        
        return all_embeddings

    def _batch_encode_images(self, images: List, batch_size: int = 16) -> List[np.ndarray]:
        """
        Batch encode multiple PIL Images for better performance.
        
        Args:
            images: List of PIL Image objects to encode
            batch_size: Number of images to process in each batch
            
        Returns:
            List of normalized embedding vectors
        """
        if not images:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            logger.debug(f"Encoding image batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
            
            try:
                batch_embeddings = self.image_model.encode(batch_images, normalize_embeddings=True)
                batch_embeddings = batch_embeddings.astype(np.float32)
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.warning(f"Batch image encoding failed: {e}")
                # Fallback to individual encoding
                for img in batch_images:
                    try:
                        embedding = self.image_model.encode([img], normalize_embeddings=True)[0]
                        embedding = embedding.astype(np.float32)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Individual image encoding failed: {e2}")
                        all_embeddings.append(np.zeros(self.image_model.get_sentence_embedding_dimension(), dtype=np.float32))
        
        return all_embeddings
    
    def _save_faiss_index(self, embeddings: np.ndarray, index_path: Path) -> bool:
        """
        Save embeddings to FAISS index.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            index_path: Path where to save the index
            
        Returns:
            True if successful, False otherwise
        """
        if not HAVE_FAISS:
            return False
            
        try:
            # Create FAISS index (Inner Product for normalized vectors)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings
            index.add(embeddings)
            
            # Save to disk
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index: {index_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save FAISS index {index_path}: {e}")
            return False
    
    def _save_numpy_index(self, embeddings: np.ndarray, index_path: Path) -> bool:
        """
        Save embeddings to NumPy file (fallback).
        
        Args:
            embeddings: Array of embeddings
            index_path: Path where to save the .npy file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            np.save(index_path, embeddings)
            logger.info(f"Saved NumPy index: {index_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save NumPy index {index_path}: {e}")
            return False
    
    def build_index(self, screenshots_dir: Path, index_dir: Path, 
                   text_batch_size: int = 32, image_batch_size: int = 16) -> Dict[str, Any]:
        """
        Build search index from screenshots directory with incremental caching and batch processing.
        
        Args:
            screenshots_dir: Directory containing screenshot images
            index_dir: Directory where to save the index files
            text_batch_size: Batch size for text embedding encoding (default: 32)
            image_batch_size: Batch size for image embedding encoding (default: 16)
            
        Returns:
            Summary dictionary with build statistics
        """
        screenshots_dir = Path(screenshots_dir)
        index_dir = Path(index_dir)
        
        # Validate input directory
        if not screenshots_dir.exists() or not screenshots_dir.is_dir():
            raise ValueError(f"Screenshots directory does not exist: {screenshots_dir}")
        
        # Create index directory
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models once
        self._load_models()
        
        # Load existing cache
        cache_path = index_dir / "cache.json"
        cache = self._load_cache(cache_path)
        
        # Find all images
        image_paths = list_images(screenshots_dir)
        if not image_paths:
            logger.warning(f"No images found in {screenshots_dir}")
            return {'num_images': 0, 'built_at': time.time(), 'skipped': 0, 'processed': 0, 'duplicates': 0}
        
        logger.info(f"Processing {len(image_paths)} images from {screenshots_dir}")
        
        # First pass: collect basic info and determine what needs processing
        processed_items = []
        texts_to_encode = []
        images_to_encode = []
        text_indices = []  # Maps position in texts_to_encode to position in final arrays
        image_indices = []  # Maps position in images_to_encode to position in final arrays
        cached_text_embeddings = {}  # Maps final index to cached embedding
        cached_image_embeddings = {}  # Maps final index to cached embedding
        
        # Perceptual hash tracking for deduplication
        phash_to_paths = {}  # Maps pHash to list of file paths
        current_run_phashes = {}  # Maps pHash to first occurrence in this run
        
        # Counters for logging
        skipped_count = 0
        processed_count = 0
        duplicate_count = 0
        
        start_time = time.time()
        
        logger.info("First pass: analyzing files and loading cached data...")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Analyzing {i+1}/{len(image_paths)}: {image_path.name}")
            
            try:
                # Compute perceptual hash for duplicate detection
                try:
                    phash = compute_phash(image_path)
                except Exception as e:
                    logger.warning(f"Failed to compute perceptual hash for {image_path.name}: {e}")
                    # Continue processing even if pHash fails
                    phash = None
                
                # Check for duplicates in current run
                if phash and phash in current_run_phashes:
                    logger.info(f"Duplicate skipped: {image_path.name} (same pHash as {current_run_phashes[phash]})")
                    duplicate_count += 1
                    
                    # Track the duplicate path for metadata
                    if phash in phash_to_paths:
                        phash_to_paths[phash].append(str(image_path))
                    else:
                        phash_to_paths[phash] = [str(image_path)]
                    
                    continue
                
                # Record this pHash for duplicate detection
                if phash:
                    current_run_phashes[phash] = image_path.name
                    phash_to_paths[phash] = [str(image_path)]
                
                # Compute file hash
                sha1_hash = file_sha1(image_path)
                
                # Get file stats
                stat = image_path.stat()
                
                # Check if we have cached data for this hash
                if sha1_hash in cache:
                    logger.debug(f"Using cached data for {image_path.name} (SHA-1: {sha1_hash[:8]}...)")
                    cached_data = cache[sha1_hash]
                    
                    # Store basic item info
                    processed_items.append({
                        'filename': image_path.name,
                        'path': str(image_path),
                        'sha1': sha1_hash,
                        'phash': phash,
                        'ocr_text': cached_data['ocr_text'],
                        'tags': cached_data['tags'],
                        'file_size': stat.st_size,
                        'modified_time': stat.st_mtime,
                        'processed_at': time.time(),
                        'from_cache': True
                    })
                    
                    # Store cached embeddings for later use
                    cached_text_embeddings[len(processed_items) - 1] = np.array(cached_data['text_vec'], dtype=np.float32)
                    cached_image_embeddings[len(processed_items) - 1] = np.array(cached_data['img_vec'], dtype=np.float32)
                    skipped_count += 1
                    
                else:
                    logger.debug(f"Needs processing: {image_path.name} (SHA-1: {sha1_hash[:8]}...)")
                    
                    # Extract text via OCR
                    ocr_text = self._extract_text_from_image(image_path)
                    
                    # Generate tags
                    tags = simple_tags(ocr_text)
                    
                    # Store basic item info
                    processed_items.append({
                        'filename': image_path.name,
                        'path': str(image_path),
                        'sha1': sha1_hash,
                        'phash': phash,
                        'ocr_text': ocr_text,
                        'tags': tags,
                        'file_size': stat.st_size,
                        'modified_time': stat.st_mtime,
                        'processed_at': time.time(),
                        'from_cache': False
                    })
                    
                    # Collect data for batch processing
                    texts_to_encode.append(ocr_text)
                    text_indices.append(len(processed_items) - 1)
                    
                    # Load image for batch processing
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            images_to_encode.append(img.copy())  # Copy since we're closing the original
                            image_indices.append(len(processed_items) - 1)
                    except Exception as e:
                        logger.warning(f"Failed to load image {image_path.name} for batch processing: {e}")
                        # Add zero embeddings as fallback
                        cached_text_embeddings[len(processed_items) - 1] = np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
                        cached_image_embeddings[len(processed_items) - 1] = np.zeros(self.image_model.get_sentence_embedding_dimension(), dtype=np.float32)
                        continue
                    
                    processed_count += 1
                    
                    # Update cache with new data (we'll add embeddings after batch processing)
                    cache[sha1_hash] = {
                        'ocr_text': ocr_text,
                        'tags': tags,
                        'text_vec': None,  # Will be filled after batch processing
                        'img_vec': None    # Will be filled after batch processing
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to process {image_path.name}: {e}")
                continue
        
        logger.info(f"Analysis complete: {processed_count} need processing, {skipped_count} from cache, {duplicate_count} duplicates skipped")
        
        # Batch process text embeddings
        text_embeddings = [None] * len(processed_items)
        if texts_to_encode:
            logger.info(f"Batch encoding {len(texts_to_encode)} texts (batch_size={text_batch_size})...")
            try:
                batch_text_embeddings = self._batch_encode_texts(texts_to_encode, text_batch_size)
                
                # Map batch results back to final positions
                for batch_idx, final_idx in enumerate(text_indices):
                    if batch_idx < len(batch_text_embeddings):
                        text_embeddings[final_idx] = batch_text_embeddings[batch_idx]
                        # Update cache with embedding
                        item = processed_items[final_idx]
                        cache[item['sha1']]['text_vec'] = batch_text_embeddings[batch_idx].tolist()
                    else:
                        logger.warning(f"Missing batch text embedding for index {batch_idx}")
            except Exception as e:
                logger.error(f"Batch text encoding failed: {e}")
                # Fallback: fill with zero vectors
                for final_idx in text_indices:
                    if text_embeddings[final_idx] is None:
                        text_embeddings[final_idx] = np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        # Add cached text embeddings
        for final_idx, embedding in cached_text_embeddings.items():
            text_embeddings[final_idx] = embedding
        
        # Batch process image embeddings
        image_embeddings = [None] * len(processed_items)
        if images_to_encode:
            logger.info(f"Batch encoding {len(images_to_encode)} images (batch_size={image_batch_size})...")
            try:
                batch_image_embeddings = self._batch_encode_images(images_to_encode, image_batch_size)
                
                # Map batch results back to final positions
                for batch_idx, final_idx in enumerate(image_indices):
                    if batch_idx < len(batch_image_embeddings):
                        image_embeddings[final_idx] = batch_image_embeddings[batch_idx]
                        # Update cache with embedding
                        item = processed_items[final_idx]
                        cache[item['sha1']]['img_vec'] = batch_image_embeddings[batch_idx].tolist()
                    else:
                        logger.warning(f"Missing batch image embedding for index {batch_idx}")
            except Exception as e:
                logger.error(f"Batch image encoding failed: {e}")
                # Fallback: fill with zero vectors
                for final_idx in image_indices:
                    if image_embeddings[final_idx] is None:
                        image_embeddings[final_idx] = np.zeros(self.image_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        # Add cached image embeddings
        for final_idx, embedding in cached_image_embeddings.items():
            image_embeddings[final_idx] = embedding
        
        # Final safety check: fill any remaining None values with zero vectors
        for i in range(len(text_embeddings)):
            if text_embeddings[i] is None:
                logger.warning(f"Filling missing text embedding at index {i} with zero vector")
                text_embeddings[i] = np.zeros(self.text_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        for i in range(len(image_embeddings)):
            if image_embeddings[i] is None:
                logger.warning(f"Filling missing image embedding at index {i} with zero vector")
                image_embeddings[i] = np.zeros(self.image_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        if not processed_items:
            logger.error("No images were successfully processed")
            return {'num_images': 0, 'built_at': time.time(), 'skipped': 0, 'processed': 0, 'duplicates': 0}
        
        # Ensure we have embeddings for all items
        missing_text = any(emb is None for emb in text_embeddings)
        missing_image = any(emb is None for emb in image_embeddings)
        
        if missing_text or missing_image:
            logger.error("Missing embeddings detected - this should not happen")
            return {'num_images': 0, 'built_at': time.time(), 'skipped': 0, 'processed': 0, 'duplicates': 0}
        
        # Log processing statistics
        logger.info(f"Processing complete: {skipped_count} skipped (cached), {processed_count} newly processed, {duplicate_count} duplicates skipped")
        
        # Convert to numpy arrays
        text_embeddings = np.array(text_embeddings, dtype=np.float32)
        image_embeddings = np.array(image_embeddings, dtype=np.float32)
        
        logger.info(f"Generated embeddings: text {text_embeddings.shape}, image {image_embeddings.shape}")
        
        # Save indexes
        text_index_path = index_dir / "text_index"
        image_index_path = index_dir / "image_index"
        
        # Try FAISS first, fallback to NumPy
        text_saved_faiss = self._save_faiss_index(text_embeddings, text_index_path.with_suffix('.faiss'))
        image_saved_faiss = self._save_faiss_index(image_embeddings, image_index_path.with_suffix('.faiss'))
        
        # Always save NumPy fallback
        text_saved_numpy = self._save_numpy_index(text_embeddings, text_index_path.with_suffix('.npy'))
        image_saved_numpy = self._save_numpy_index(image_embeddings, image_index_path.with_suffix('.npy'))
        
        # Save updated cache
        self._save_cache(cache, cache_path)
        
        # Create metadata
        built_at = time.time()
        metadata = {
            'built_at': built_at,
            'num_images': len(processed_items),
            'screenshots_dir': str(screenshots_dir),
            'have_faiss': text_saved_faiss and image_saved_faiss,
            'have_numpy': text_saved_numpy and image_saved_numpy,
            'text_embedding_dim': text_embeddings.shape[1],
            'image_embedding_dim': image_embeddings.shape[1],
            'processing_time': time.time() - start_time,
            'skipped_count': skipped_count,
            'processed_count': processed_count,
            'duplicate_count': duplicate_count,
            'cache_entries': len(cache),
            'phash_groups': phash_to_paths,  # Track duplicate groups by pHash
            'items': processed_items
        }
        
        # Save metadata
        meta_path = index_dir / "meta.json"
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata: {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
        
        logger.info(f"Index building completed in {metadata['processing_time']:.1f}s")
        logger.info(f"Successfully indexed {len(processed_items)} images ({skipped_count} from cache, {processed_count} newly processed, {duplicate_count} duplicates skipped)")
        
        return {
            'num_images': len(processed_items),
            'built_at': built_at,
            'processing_time': metadata['processing_time'],
            'have_faiss': metadata['have_faiss'],
            'have_numpy': metadata['have_numpy'],
            'skipped': skipped_count,
            'processed': processed_count,
            'duplicates': duplicate_count,
            'cache_entries': len(cache)
        }


def load_index_summary(index_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load index summary information from metadata.
    
    Args:
        index_dir: Directory containing the index files
        
    Returns:
        Summary dictionary or None if not found/invalid
    """
    index_dir = Path(index_dir)
    meta_path = index_dir / "meta.json"
    
    if not meta_path.exists():
        logger.warning(f"Index metadata not found: {meta_path}")
        return None
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Return summary subset
        return {
            'num_images': metadata.get('num_images', 0),
            'built_at': metadata.get('built_at', 0),
            'screenshots_dir': metadata.get('screenshots_dir', ''),
            'have_faiss': metadata.get('have_faiss', False),
            'have_numpy': metadata.get('have_numpy', False),
            'processing_time': metadata.get('processing_time', 0),
            'text_embedding_dim': metadata.get('text_embedding_dim', 0),
            'image_embedding_dim': metadata.get('image_embedding_dim', 0),
            'skipped_count': metadata.get('skipped_count', 0),
            'processed_count': metadata.get('processed_count', 0),
            'duplicate_count': metadata.get('duplicate_count', 0),
            'cache_entries': metadata.get('cache_entries', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to load index summary: {e}")
        return None


def build_index(screenshots_dir: Path, index_dir: Path, 
               text_batch_size: int = 32, image_batch_size: int = 16) -> Dict[str, Any]:
    """
    Convenience function to build index with batch processing.
    
    Args:
        screenshots_dir: Directory containing screenshots
        index_dir: Directory to save index files
        text_batch_size: Batch size for text embedding encoding (default: 32)
        image_batch_size: Batch size for image embedding encoding (default: 16)
        
    Returns:
        Build summary dictionary
    """
    indexer = ImageIndexer()
    return indexer.build_index(screenshots_dir, index_dir, text_batch_size, image_batch_size)


def main():
    """CLI entry point for headless indexing."""
    parser = argparse.ArgumentParser(description='Build Smart Screenshot Search index')
    parser.add_argument('--folder', '--screenshots', required=True,
                       help='Directory containing screenshot images')
    parser.add_argument('--index', '--output', required=True,
                       help='Directory to save index files')
    parser.add_argument('--text-batch-size', type=int, default=32,
                       help='Batch size for text embedding encoding (default: 32)')
    parser.add_argument('--image-batch-size', type=int, default=16,
                       help='Batch size for image embedding encoding (default: 16)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    screenshots_dir = Path(args.folder)
    index_dir = Path(args.index)
    
    try:
        logger.info(f"Building index from {screenshots_dir} to {index_dir}")
        logger.info(f"Batch sizes: text={args.text_batch_size}, image={args.image_batch_size}")
        summary = build_index(screenshots_dir, index_dir, args.text_batch_size, args.image_batch_size)
        
        print(f"\n=== Index Build Complete ===")
        print(f"Images processed: {summary['num_images']}")
        print(f"Newly processed: {summary.get('processed', 0)}")
        print(f"Skipped (cached): {summary.get('skipped', 0)}")
        print(f"Duplicates skipped: {summary.get('duplicates', 0)}")
        print(f"Cache entries: {summary.get('cache_entries', 0)}")
        print(f"Processing time: {summary.get('processing_time', 0):.1f}s")
        print(f"FAISS available: {summary.get('have_faiss', False)}")
        print(f"NumPy fallback: {summary.get('have_numpy', False)}")
        print(f"Index directory: {index_dir}")
        
        # Load and verify summary
        summary_check = load_index_summary(index_dir)
        if summary_check:
            print(f"✓ Index metadata verified")
        else:
            print(f"⚠ Could not verify index metadata")
            
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())