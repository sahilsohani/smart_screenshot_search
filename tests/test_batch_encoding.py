"""
Unit tests for batch encoding functionality in Smart Screenshot Search.

Tests that batch encoding produces identical results to individual encoding
within floating point tolerance.
"""

import numpy as np
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from typing import List

import sys
import os
# Add the parent directory to sys.path to import smartshot modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smartshot.indexer import ImageIndexer


class TestBatchEncoding(unittest.TestCase):
    """Test batch encoding vs individual encoding equivalence."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.indexer = ImageIndexer()
        self.indexer._load_models()  # Load models once for all tests
        
        # Create test texts
        self.test_texts = [
            "Hello world",
            "Receipt Total: $45.99",
            "",  # Empty text
            "Flight DL123 Gate A5",
            "This is a longer text with multiple words and sentences. It should test the encoding properly.",
            "   ",  # Whitespace only
            "Special chars: !@#$%^&*()",
            "Numbers: 123 456 789"
        ]
        
        # Create test images
        self.test_images = []
        self.temp_files = []
        
        # Create various test images - all in RGB mode for CLIP compatibility
        for i, (size, color) in enumerate([
            ((100, 100), (255, 0, 0)),    # Red
            ((200, 150), (0, 0, 255)),    # Blue
            ((50, 75), (0, 255, 0)),      # Green
            ((300, 200), (255, 255, 0)),  # Yellow
            ((10, 10), (0, 0, 0))         # Black (increased size for stability)
        ]):
            img = Image.new('RGB', size, color)
            self.test_images.append(img)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close any opened images
        for img in self.test_images:
            if hasattr(img, 'close'):
                img.close()
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_batch_text_encoding_equivalence(self):
        """Test that batch text encoding produces identical results to individual encoding."""
        print("Testing batch text encoding equivalence...")
        
        # Individual encoding
        individual_embeddings = []
        for text in self.test_texts:
            embedding = self.indexer._create_text_embedding(text)
            individual_embeddings.append(embedding)
        
        # Batch encoding with different batch sizes
        for batch_size in [1, 3, 5, 10]:
            with self.subTest(batch_size=batch_size):
                batch_embeddings = self.indexer._batch_encode_texts(self.test_texts, batch_size)
                
                self.assertEqual(len(batch_embeddings), len(individual_embeddings),
                               f"Batch size {batch_size}: Different number of embeddings")
                
                for i, (individual, batch) in enumerate(zip(individual_embeddings, batch_embeddings)):
                    # Check shapes match
                    self.assertEqual(individual.shape, batch.shape,
                                   f"Batch size {batch_size}, text {i}: Shape mismatch")
                    
                    # Check values are close (within floating point tolerance)
                    np.testing.assert_allclose(individual, batch, rtol=1e-6, atol=1e-6,
                                             err_msg=f"Batch size {batch_size}, text {i}: Values differ")
                    
                    # Check data types match
                    self.assertEqual(individual.dtype, batch.dtype,
                                   f"Batch size {batch_size}, text {i}: Dtype mismatch")
        
        print("✓ Batch text encoding equivalence test passed")
    
    def test_batch_image_encoding_equivalence(self):
        """Test that batch image encoding produces identical results to individual encoding."""
        print("Testing batch image encoding equivalence...")
        
        # Individual encoding using the same method as the indexer
        individual_embeddings = []
        for img in self.test_images:
            # Ensure image is RGB (same as indexer does)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            embedding = self.indexer.image_model.encode([img], normalize_embeddings=True)[0]
            embedding = embedding.astype(np.float32)
            individual_embeddings.append(embedding)
        
        # Batch encoding with different batch sizes
        for batch_size in [1, 2, 3, 10]:
            with self.subTest(batch_size=batch_size):
                batch_embeddings = self.indexer._batch_encode_images(self.test_images, batch_size)
                
                self.assertEqual(len(batch_embeddings), len(individual_embeddings),
                               f"Batch size {batch_size}: Different number of embeddings")
                
                for i, (individual, batch) in enumerate(zip(individual_embeddings, batch_embeddings)):
                    # Check shapes match
                    self.assertEqual(individual.shape, batch.shape,
                                   f"Batch size {batch_size}, image {i}: Shape mismatch")
                    
                    # Check values are close (within floating point tolerance)
                    np.testing.assert_allclose(individual, batch, rtol=1e-5, atol=1e-5,
                                             err_msg=f"Batch size {batch_size}, image {i}: Values differ")
                    
                    # Check data types match
                    self.assertEqual(individual.dtype, batch.dtype,
                                   f"Batch size {batch_size}, image {i}: Dtype mismatch")
        
        print("✓ Batch image encoding equivalence test passed")
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        print("Testing empty batch handling...")
        
        # Test empty text list
        result = self.indexer._batch_encode_texts([])
        self.assertEqual(len(result), 0, "Empty text batch should return empty list")
        
        # Test empty image list
        result = self.indexer._batch_encode_images([])
        self.assertEqual(len(result), 0, "Empty image batch should return empty list")
        
        print("✓ Empty batch handling test passed")
    
    def test_batch_order_preservation(self):
        """Test that batch processing preserves order correctly."""
        print("Testing batch order preservation...")
        
        # Create texts with distinguishable embeddings
        ordered_texts = [f"Text number {i}" for i in range(10)]
        
        # Get individual embeddings in order
        individual_embeddings = []
        for text in ordered_texts:
            embedding = self.indexer._create_text_embedding(text)
            individual_embeddings.append(embedding)
        
        # Test with different batch sizes that don't divide evenly
        for batch_size in [3, 7]:  # These don't divide 10 evenly
            with self.subTest(batch_size=batch_size):
                batch_embeddings = self.indexer._batch_encode_texts(ordered_texts, batch_size)
                
                self.assertEqual(len(batch_embeddings), len(individual_embeddings),
                               f"Batch size {batch_size}: Length mismatch")
                
                for i, (individual, batch) in enumerate(zip(individual_embeddings, batch_embeddings)):
                    np.testing.assert_allclose(individual, batch, rtol=1e-6, atol=1e-6,
                                             err_msg=f"Batch size {batch_size}: Order not preserved at position {i}")
        
        print("✓ Batch order preservation test passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("Testing edge cases...")
        
        # Test single item batches
        single_text = ["Single text item"]
        result = self.indexer._batch_encode_texts(single_text, batch_size=1)
        self.assertEqual(len(result), 1, "Single text batch failed")
        
        single_image = [self.test_images[0]]
        result = self.indexer._batch_encode_images(single_image, batch_size=1)
        self.assertEqual(len(result), 1, "Single image batch failed")
        
        # Test batch size larger than input
        result = self.indexer._batch_encode_texts(["text1", "text2"], batch_size=10)
        self.assertEqual(len(result), 2, "Large batch size failed")
        
        result = self.indexer._batch_encode_images([self.test_images[0]], batch_size=10)
        self.assertEqual(len(result), 1, "Large image batch size failed")
        
        print("✓ Edge cases test passed")
    
    def test_performance_benchmark(self):
        """Basic performance comparison between individual and batch encoding."""
        print("Running performance benchmark...")
        
        import time
        
        # Create a reasonable number of test items
        benchmark_texts = [f"Benchmark text {i} with some content" for i in range(20)]
        benchmark_images = [self.test_images[0]] * 10  # Reuse same image
        
        # Time individual text encoding
        start_time = time.time()
        individual_text_results = []
        for text in benchmark_texts:
            embedding = self.indexer._create_text_embedding(text)
            individual_text_results.append(embedding)
        individual_text_time = time.time() - start_time
        
        # Time batch text encoding
        start_time = time.time()
        batch_text_results = self.indexer._batch_encode_texts(benchmark_texts, batch_size=8)
        batch_text_time = time.time() - start_time
        
        # Time individual image encoding (fewer items due to slower processing)
        start_time = time.time()
        individual_image_results = []
        for img in benchmark_images:
            # Ensure RGB mode for consistency
            if img.mode != 'RGB':
                img = img.convert('RGB')
            embedding = self.indexer.image_model.encode([img], normalize_embeddings=True)[0]
            individual_image_results.append(embedding.astype(np.float32))
        individual_image_time = time.time() - start_time
        
        # Time batch image encoding
        start_time = time.time()
        batch_image_results = self.indexer._batch_encode_images(benchmark_images, batch_size=4)
        batch_image_time = time.time() - start_time
        
        print(f"Text encoding - Individual: {individual_text_time:.3f}s, Batch: {batch_text_time:.3f}s")
        print(f"Image encoding - Individual: {individual_image_time:.3f}s, Batch: {batch_image_time:.3f}s")
        
        # Verify results are still equivalent
        for i, (individual, batch) in enumerate(zip(individual_text_results, batch_text_results)):
            np.testing.assert_allclose(individual, batch, rtol=1e-6, atol=1e-6,
                                     err_msg=f"Performance test: Text {i} differs")
        
        for i, (individual, batch) in enumerate(zip(individual_image_results, batch_image_results)):
            np.testing.assert_allclose(individual, batch, rtol=1e-5, atol=1e-5,
                                     err_msg=f"Performance test: Image {i} differs")
        
        print("✓ Performance benchmark completed (results verified identical)")


def run_tests():
    """Run all batch encoding tests."""
    print("=" * 60)
    print("SMART SCREENSHOT SEARCH - BATCH ENCODING TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchEncoding)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Ran {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)