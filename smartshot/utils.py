"""
Utility functions for Smart Screenshot Search.

This module provides core utilities for file handling, image processing,
platform detection, and content analysis.
"""

import hashlib
import os
import platform
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageOps


# Supported image extensions
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def list_images(folder: Union[str, Path]) -> List[Path]:
    """
    List all supported image files in a folder (non-recursive).
    
    Args:
        folder: Path to the folder to search
        
    Returns:
        List of Path objects for supported image files
        
    Example:
        >>> images = list_images("/path/to/screenshots")
        >>> len(images)
        42
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    
    images = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMG_EXTS:
            images.append(file_path)
    
    return sorted(images)  # Consistent ordering


def file_sha1(path: Union[str, Path]) -> str:
    """
    Compute SHA-1 hash of a file using streaming to handle large files.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        SHA-1 hash as hexadecimal string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file can't be read
        
    Example:
        >>> file_sha1("screenshot.png")
        'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha1_hash = hashlib.sha1()
    
    try:
        with open(file_path, 'rb') as f:
            # Read in 64KB chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(65536), b""):
                sha1_hash.update(chunk)
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    
    return sha1_hash.hexdigest()


def preview_image(path: Union[str, Path], max_px: int = 800) -> Image.Image:
    """
    Load and resize an image for preview, maintaining aspect ratio.
    
    Args:
        path: Path to the image file
        max_px: Maximum dimension (width or height) in pixels
        
    Returns:
        PIL Image object, resized if necessary
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        PIL.UnidentifiedImageError: If the file is not a valid image
        
    Example:
        >>> img = preview_image("large_screenshot.png", max_px=400)
        >>> img.size
        (400, 300)  # Maintains aspect ratio
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Apply EXIF orientation if present
            img = ImageOps.exif_transpose(img)
            
            # Resize if image is larger than max_px in any dimension
            width, height = img.size
            if max(width, height) > max_px:
                # Calculate new size maintaining aspect ratio
                if width > height:
                    new_width = max_px
                    new_height = int(height * max_px / width)
                else:
                    new_height = max_px
                    new_width = int(width * max_px / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Return a copy since we're closing the original
            return img.copy()
            
    except Exception as e:
        raise type(e)(f"Error processing image {file_path}: {e}")


def pick_default_screenshot_dir() -> Optional[Path]:
    """
    Pick a sensible default screenshot directory based on the operating system.
    
    Returns:
        Path to likely screenshot directory, or None if not found
        
    Example:
        >>> default_dir = pick_default_screenshot_dir()
        >>> print(default_dir)
        /Users/username/Desktop  # on macOS
    """
    system = platform.system().lower()
    home = Path.home()
    
    # Define potential screenshot directories for each OS
    candidates = []
    
    if system == 'darwin':  # macOS
        candidates = [
            home / 'Desktop',
            home / 'Screenshots',
            home / 'Pictures' / 'Screenshots',
            home / 'Documents' / 'Screenshots'
        ]
    elif system == 'windows':
        candidates = [
            home / 'Desktop',
            home / 'Pictures' / 'Screenshots',
            home / 'Documents' / 'Screenshots',
            home / 'OneDrive' / 'Desktop',
            home / 'OneDrive' / 'Pictures' / 'Screenshots'
        ]
    elif system == 'linux':
        candidates = [
            home / 'Desktop',
            home / 'Pictures' / 'Screenshots',
            home / 'Pictures',
            home / 'Documents' / 'Screenshots'
        ]
    
    # Return the first existing directory with images
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it contains any images
            if list_images(candidate):
                return candidate
    
    # Fallback: return first existing directory even without images
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    
    # Last resort: return Desktop or home directory
    desktop = home / 'Desktop'
    if desktop.exists():
        return desktop
    
    return home if home.exists() else None


def simple_tags(ocr_text: str) -> List[str]:
    """
    Generate simple tags based on OCR text content using keyword matching.
    
    Args:
        ocr_text: Text extracted from image via OCR
        
    Returns:
        Sorted list of tags
        
    Example:
        >>> simple_tags("Total: $45.99 Thank you for your purchase")
        ['receipt', 'shopping']
    """
    if not ocr_text:
        return []
    
    text_lower = ocr_text.lower()
    tags = set()
    
    # Receipt indicators
    receipt_keywords = {
        'total', 'subtotal', 'tax', 'receipt', 'purchase', 'payment',
        'visa', 'mastercard', 'card', 'cash', 'change', 'thank you',
        '$', '€', '£', '¥', 'usd', 'eur', 'gbp', 'price', 'amount'
    }
    if any(keyword in text_lower for keyword in receipt_keywords):
        tags.add('receipt')
        tags.add('shopping')
    
    # Travel indicators
    travel_keywords = {
        'flight', 'airline', 'airport', 'boarding', 'gate', 'seat',
        'departure', 'arrival', 'terminal', 'baggage', 'passport',
        'visa', 'hotel', 'booking', 'reservation', 'check-in',
        'itinerary', 'ticket', 'confirmation'
    }
    if any(keyword in text_lower for keyword in travel_keywords):
        tags.add('travel')
    
    # Food/restaurant indicators
    food_keywords = {
        'menu', 'restaurant', 'order', 'delivery', 'pickup', 'pizza',
        'burger', 'coffee', 'tea', 'breakfast', 'lunch', 'dinner',
        'appetizer', 'dessert', 'beverage', 'drink', 'food', 'kitchen'
    }
    if any(keyword in text_lower for keyword in food_keywords):
        tags.add('food')
    
    # Webpage/browser indicators
    web_keywords = {
        'http', 'https', 'www', '.com', '.org', '.net', 'browser',
        'chrome', 'firefox', 'safari', 'edge', 'url', 'website',
        'login', 'password', 'email', 'username', 'sign in', 'sign up'
    }
    if any(keyword in text_lower for keyword in web_keywords):
        tags.add('webpage')
        tags.add('browser')
    
    # Communication indicators
    comm_keywords = {
        'message', 'chat', 'email', 'text', 'sms', 'whatsapp',
        'telegram', 'slack', 'teams', 'zoom', 'meet', 'call'
    }
    if any(keyword in text_lower for keyword in comm_keywords):
        tags.add('communication')
    
    # Document indicators
    doc_keywords = {
        'document', 'pdf', 'report', 'invoice', 'contract', 'agreement',
        'form', 'application', 'certificate', 'license', 'permit'
    }
    if any(keyword in text_lower for keyword in doc_keywords):
        tags.add('document')
    
    # Code/development indicators
    code_keywords = {
        'function', 'class', 'import', 'def', 'return', 'if', 'else',
        'for', 'while', 'try', 'except', 'github', 'git', 'commit',
        'pull request', 'merge', 'branch', 'repository'
    }
    if any(keyword in text_lower for keyword in code_keywords):
        tags.add('code')
        tags.add('development')
    
    # Generic content type detection
    if len(text_lower) > 100:
        tags.add('text-heavy')
    
    # Check for numbers (could indicate data/analytics)
    import re
    if re.search(r'\d+', text_lower):
        tags.add('contains-numbers')
    
    return sorted(list(tags))


if __name__ == "__main__":
    """Self-tests and examples"""
    import tempfile
    import shutil
    from io import BytesIO
    
    print("=== Smart Screenshot Search Utils Tests ===\n")
    
    # Test 1: IMG_EXTS and list_images
    print("1. Testing list_images...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some test files
        (temp_path / "image1.png").touch()
        (temp_path / "image2.jpg").touch()
        (temp_path / "image3.JPEG").touch()  # Test case sensitivity
        (temp_path / "not_image.txt").touch()
        (temp_path / "image4.webp").touch()
        
        images = list_images(temp_path)
        print(f"   Found {len(images)} images: {[img.name for img in images]}")
        assert len(images) == 4, f"Expected 4 images, got {len(images)}"
    
    # Test 2: file_sha1
    print("\n2. Testing file_sha1...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        test_content = b"Hello, Smart Screenshot Search!"
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        hash_result = file_sha1(temp_file_path)
        expected_hash = hashlib.sha1(test_content).hexdigest()
        print(f"   Hash: {hash_result}")
        assert hash_result == expected_hash, "SHA-1 hash mismatch"
        print("   ✓ SHA-1 calculation correct")
    finally:
        os.unlink(temp_file_path)
    
    # Test 3: preview_image
    print("\n3. Testing preview_image...")
    # Create a simple test image
    test_img = Image.new('RGB', (1000, 600), color='red')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
        test_img.save(temp_img.name)
        temp_img_path = temp_img.name
    
    try:
        preview = preview_image(temp_img_path, max_px=400)
        print(f"   Original size: (1000, 600)")
        print(f"   Preview size: {preview.size}")
        # Should maintain aspect ratio: 1000/600 = 400/240
        assert preview.size == (400, 240), f"Unexpected preview size: {preview.size}"
        print("   ✓ Aspect ratio maintained correctly")
    finally:
        os.unlink(temp_img_path)
    
    # Test 4: pick_default_screenshot_dir
    print("\n4. Testing pick_default_screenshot_dir...")
    default_dir = pick_default_screenshot_dir()
    print(f"   Default screenshot directory: {default_dir}")
    if default_dir:
        print(f"   Directory exists: {default_dir.exists()}")
    
    # Test 5: simple_tags
    print("\n5. Testing simple_tags...")
    
    test_cases = [
        ("Total: $45.99 Thank you for your purchase", "Receipt"),
        ("Flight DL123 Gate A5 Boarding Pass", "Travel"),
        ("Pizza Delivery Order #12345 Pepperoni Large", "Food"),
        ("https://github.com/user/repo Pull Request #42", "Code/Web"),
        ("Meeting invite via email calendar@company.com", "Communication"),
        ("", "Empty text")
    ]
    
    for text, description in test_cases:
        tags = simple_tags(text)
        print(f"   {description}: {tags}")
    
    print("\n=== All tests completed! ===")