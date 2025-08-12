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

from .utils import list_images, file_sha1, simple_tags

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
    
    def _process_single_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single image: compute hash, OCR, tags, and embeddings.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image metadata and embeddings, or None if failed
        """
        try:
            # Compute file hash
            sha1_hash = file_sha1(image_path)
            
            # Extract text via OCR
            ocr_text = self._extract_text_from_image(image_path)
            
            # Generate tags
            tags = simple_tags(ocr_text)
            
            # Create embeddings
            text_embedding = self._create_text_embedding(ocr_text)
            image_embedding = self._create_image_embedding(image_path)
            
            # Get file stats
            stat = image_path.stat()
            
            return {
                'filename': image_path.name,
                'path': str(image_path),
                'sha1': sha1_hash,
                'ocr_text': ocr_text,
                'tags': tags,
                'text_embedding': text_embedding,
                'image_embedding': image_embedding,
                'file_size': stat.st_size,
                'modified_time': stat.st_mtime,
                'processed_at': time.time()
            }
            
        except Exception as e:
            logger.warning(f"Failed to process {image_path.name}: {e}")
            return None
    
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
    
    def build_index(self, screenshots_dir: Path, index_dir: Path) -> Dict[str, Any]:
        """
        Build search index from screenshots directory.
        
        Args:
            screenshots_dir: Directory containing screenshot images
            index_dir: Directory where to save the index files
            
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
        
        # Find all images
        image_paths = list_images(screenshots_dir)
        if not image_paths:
            logger.warning(f"No images found in {screenshots_dir}")
            return {'num_images': 0, 'built_at': time.time()}
        
        logger.info(f"Processing {len(image_paths)} images from {screenshots_dir}")
        
        # Process all images
        processed_items = []
        text_embeddings = []
        image_embeddings = []
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self._process_single_image(image_path)
            if result is not None:
                processed_items.append({
                    'filename': result['filename'],
                    'path': result['path'],
                    'sha1': result['sha1'],
                    'ocr_text': result['ocr_text'],
                    'tags': result['tags'],
                    'file_size': result['file_size'],
                    'modified_time': result['modified_time'],
                    'processed_at': result['processed_at']
                })
                text_embeddings.append(result['text_embedding'])
                image_embeddings.append(result['image_embedding'])
        
        if not processed_items:
            logger.error("No images were successfully processed")
            return {'num_images': 0, 'built_at': time.time()}
        
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
        logger.info(f"Successfully indexed {len(processed_items)} images")
        
        return {
            'num_images': len(processed_items),
            'built_at': built_at,
            'processing_time': metadata['processing_time'],
            'have_faiss': metadata['have_faiss'],
            'have_numpy': metadata['have_numpy']
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
            'image_embedding_dim': metadata.get('image_embedding_dim', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to load index summary: {e}")
        return None


def build_index(screenshots_dir: Path, index_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to build index.
    
    Args:
        screenshots_dir: Directory containing screenshots
        index_dir: Directory to save index files
        
    Returns:
        Build summary dictionary
    """
    indexer = ImageIndexer()
    return indexer.build_index(screenshots_dir, index_dir)


def main():
    """CLI entry point for headless indexing."""
    parser = argparse.ArgumentParser(description='Build Smart Screenshot Search index')
    parser.add_argument('--folder', '--screenshots', required=True,
                       help='Directory containing screenshot images')
    parser.add_argument('--index', '--output', required=True,
                       help='Directory to save index files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    screenshots_dir = Path(args.folder)
    index_dir = Path(args.index)
    
    try:
        logger.info(f"Building index from {screenshots_dir} to {index_dir}")
        summary = build_index(screenshots_dir, index_dir)
        
        print(f"\n=== Index Build Complete ===")
        print(f"Images processed: {summary['num_images']}")
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