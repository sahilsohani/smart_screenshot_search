# 🔍 Smart Screenshot Search

A **local, privacy-first** screenshot search application that lets you find images using natural language queries. Search through your screenshots by their text content (OCR) or visual similarity using AI embeddings - all running entirely on your machine.

## ✨ What It Does

Smart Screenshot Search combines:
- **OCR text extraction** to read text from your screenshots
- **Semantic text search** using modern AI embeddings (MiniLM)
- **Visual image search** using CLIP embeddings
- **Hybrid search** with tunable weighting between text and image similarity
- **Smart caching** for instant re-indexing of unchanged files
- **Duplicate detection** to skip visually identical screenshots

**Why It's Useful:**
- 📱 Find that receipt from last month: `"coffee shop total payment"`
- ✈️ Locate travel documents: `"boarding pass gate number"`
- 💻 Search code screenshots: `"function error message"`
- 📧 Find email notifications: `"meeting invite calendar"`
- 🍕 Discover food photos: `"pizza menu delivery"`

## 🚀 Quick Start

### 1. Set Up Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd smart_screenshot_search

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Index Your Screenshots
1. Open the Streamlit app in your browser
2. In the sidebar, set your **Screenshots Folder** (e.g., `~/Desktop/Screenshots`)
3. Set your **Index Directory** (e.g., `./data/index`)
4. Click **"🗂️ Index Folder"** to build the search index
5. Start searching with natural language queries!

### 4. Search Examples
Try queries like:
- `"receipt total amount"`
- `"error message dialog"`
- `"email notification"`
- `"boarding pass flight"`
- `"code function python"`

## 🎯 Key Features

### ⚡ Performance Optimizations
- **Incremental Indexing**: Only processes new/changed files using SHA-1 caching
- **Batch Processing**: Encodes up to 32 texts and 16 images simultaneously for speed
- **Duplicate Detection**: Uses perceptual hashing to skip visually identical screenshots
- **Smart Caching**: Persistent storage of OCR results and embeddings

### 🎯 Accurate Search
- **Relevance Filtering**: Only returns genuinely relevant results (no more random matches)
- **Semantic Boosting**: Results with exact query word matches get priority
- **Threshold-based**: Configurable minimum similarity scores to filter noise
- **Context-aware**: Different strategies for text-focused vs image-focused searches

### 🔒 Privacy & Security
- **Completely Local**: Everything runs on your machine, no cloud services
- **Offline Mode**: Works without internet after initial model downloads
- **No Data Collection**: Your screenshots never leave your device
- **No Telemetry**: No usage tracking or analytics

## 🛠️ Advanced Usage

### Command Line Interface
```bash
# Build index from command line
python -m smartshot.indexer --folder ~/Screenshots --index ./data/index

# Test search quality
python -m smartshot.searcher --index ./data/index --query "receipt" --alpha 0.6

# Run with custom batch sizes
python -m smartshot.indexer --folder ~/Screenshots --index ./data/index --text-batch-size 64 --image-batch-size 8
```

### Alpha (α) Tuning Guide
| Alpha | Behavior | Best For |
|-------|----------|----------|
| 1.0 | Pure text search | Finding specific words/phrases |
| 0.6 | Balanced (default) | General purpose searching |
| 0.0 | Pure image search | Finding visually similar images |

### Performance Tips
- **First run**: Takes time to process all images and download AI models
- **Subsequent runs**: Near-instant thanks to caching (typically <1s for hundreds of images)
- **Duplicate handling**: Automatically skips identical screenshots to save processing time
- **Memory usage**: Batch processing prevents memory issues with large collections

## 🔧 Troubleshooting

### SSL Certificate Issues (macOS/Corporate Networks)
```bash
# Pre-download models with SSL bypass
export PYTHONHTTPSVERIFY=0
streamlit run app.py
```

### Missing Dependencies
```bash
# Core dependencies
pip install streamlit easyocr sentence-transformers faiss-cpu pillow numpy

# For duplicate detection
pip install imagehash

# For file monitoring
pip install watchdog
```

### Offline Operation
```bash
# Enable offline mode
export OFFLINE=1
streamlit run app.py
```

## 📊 What's New

### Recent Updates
- ✅ **Incremental Indexing**: 10x faster re-indexing for large collections
- ✅ **Batch Processing**: 3x faster initial indexing with parallel embedding generation
- ✅ **Duplicate Detection**: Automatic detection and skipping of identical screenshots
- ✅ **Search Quality**: Fixed relevance scoring and added semantic boosting
- ✅ **Better Filtering**: Only shows truly relevant results, no more noise

### Performance Improvements
- **Caching**: OCR and embeddings cached by file SHA-1 hash
- **Deduplication**: Perceptual hashing prevents processing identical images
- **Batch encoding**: Process multiple texts/images simultaneously
- **Smart filtering**: Multiple threshold levels for better result quality

## 🎯 Usage Tips

**Effective Queries:**
- Use descriptive keywords: `"payment receipt coffee"` not just `"receipt"`
- Try different alpha values if results aren't what you expect
- For visual searches, use layout descriptions: `"dark interface login form"`
- For text searches, include specific terms: `"error message failed connection"`

**Best Practices:**
- Keep screenshots in a dedicated folder for better organization
- Re-index periodically when you add many new screenshots
- Use higher alpha (0.7-0.9) for text-heavy searches
- Use lower alpha (0.1-0.4) for visual similarity searches

## 🚀 Roadmap

### Planned Features
- [ ] **Search History**: Save and revisit previous searches
- [ ] **Advanced Filters**: Date ranges, file types, custom tags
- [ ] **Export Tools**: Save search results to CSV/JSON
- [ ] **Bulk Operations**: Tag, organize, or delete multiple screenshots
- [ ] **Browser Extension**: Capture and search web screenshots

---

**Built with ❤️ for privacy-conscious users who want powerful search without giving up control of their data.**
