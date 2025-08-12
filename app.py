"""
Smart Screenshot Search - Streamlit Application

A local, privacy-first screenshot search app using OCR text extraction
and multimodal embeddings for semantic search.
"""

import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# Import our custom modules
try:
    from smartshot.indexer import build_index, load_index_summary
    from smartshot.searcher import HybridSearcher, IndexNotFoundError, InvalidIndexError
    from smartshot.utils import pick_default_screenshot_dir, list_images, preview_image
except ImportError as e:
    st.error(f"Failed to import smartshot modules: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Smart Screenshot Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_timestamp(timestamp: float) -> str:
    """Format timestamp as human readable date."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


@st.cache_resource
def load_searcher(index_dir: str) -> Optional[HybridSearcher]:
    """
    Load and cache the HybridSearcher instance.
    
    Args:
        index_dir: Path to index directory
        
    Returns:
        HybridSearcher instance or None if failed
    """
    try:
        return HybridSearcher(index_dir)
    except (IndexNotFoundError, InvalidIndexError, FileNotFoundError) as e:
        st.session_state.searcher_error = str(e)
        return None
    except Exception as e:
        st.session_state.searcher_error = f"Unexpected error loading searcher: {e}"
        return None


def build_index_with_progress(screenshots_dir: Path, index_dir: Path):
    """
    Build index with progress display.
    
    Args:
        screenshots_dir: Directory containing screenshots
        index_dir: Directory to save index files
    """
    # Check if there are images to index
    images = list_images(screenshots_dir)
    if not images:
        st.error(f"No supported images found in {screenshots_dir}")
        return
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"Indexing {len(images)} images...")
        
        # Build the index
        with st.spinner("Building search index..."):
            summary = build_index(screenshots_dir, index_dir)
        
        progress_bar.progress(100)
        
        # Show results
        if summary['num_images'] > 0:
            st.success(f"✅ Successfully indexed {summary['num_images']} images!")
            st.info(f"Processing time: {summary.get('processing_time', 0):.1f}s")
            
            # Clear the searcher cache to reload with new index
            load_searcher.clear()
            
            # Rerun to refresh the UI
            st.rerun()
        else:
            st.error("No images were successfully processed")
            
    except Exception as e:
        st.error(f"Indexing failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()


def render_search_result(result: dict, col):
    """
    Render a single search result in a column.
    
    Args:
        result: Search result dictionary
        col: Streamlit column to render in
    """
    with col:
        # Load and display image preview
        try:
            img_path = Path(result['path'])
            if img_path.exists():
                # Use our preview function to get properly sized image
                preview_img = preview_image(img_path, max_px=300)
                col.image(preview_img, use_column_width=True)
            else:
                col.error("🖼️ Image not found")
        except Exception as e:
            col.error(f"🖼️ Error loading image: {e}")
        
        # Filename and score
        col.write(f"**{result['filename']}**")
        
        # Score with color coding
        score = result['score']
        if score > 0.8:
            col.success(f"Score: {score:.3f}")
        elif score > 0.5:
            col.info(f"Score: {score:.3f}")
        else:
            col.warning(f"Score: {score:.3f}")
        
        # Tags if available
        if result.get('tags'):
            tags_str = ", ".join(result['tags'])
            col.caption(f"🏷️ {tags_str}")
        
        # Expandable details
        with col.expander("📋 Details", expanded=False):
            st.write(f"**Path:** `{result['path']}`")
            
            # Scores breakdown
            st.write("**Score Breakdown:**")
            st.write(f"- Combined: {result['score']:.3f}")
            st.write(f"- Text: {result['text_score']:.3f}")
            st.write(f"- Image: {result['image_score']:.3f}")
            
            # File info
            if result.get('file_size'):
                st.write(f"**Size:** {format_file_size(result['file_size'])}")
            
            if result.get('modified_time'):
                st.write(f"**Modified:** {format_timestamp(result['modified_time'])}")
            
            # OCR snippet
            if result.get('snippet'):
                st.write("**Text Content:**")
                st.code(result['snippet'], language=None)
            else:
                st.write("**Text Content:** *(No text detected)*")


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Smart Screenshot Search")
    st.caption("Local, privacy-first screenshot search using OCR and AI embeddings")
    
    # Initialize session state
    if 'searcher_error' not in st.session_state:
        st.session_state.searcher_error = None
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Screenshot folder selection
    st.sidebar.subheader("📁 Folders")
    
    # Get default screenshot directory
    default_screenshots_dir = pick_default_screenshot_dir()
    default_screenshots_str = str(default_screenshots_dir) if default_screenshots_dir else ""
    
    screenshots_folder = st.sidebar.text_input(
        "Screenshots Folder",
        value=default_screenshots_str,
        placeholder="/path/to/your/screenshots",
        help="Directory containing your screenshot images"
    )
    
    index_folder = st.sidebar.text_input(
        "Index Directory", 
        value="./data/index",
        placeholder="./data/index",
        help="Directory to store search index files"
    )
    
    # Search parameters
    st.sidebar.subheader("🎛️ Search Parameters")
    
    top_k = st.sidebar.slider(
        "Results Count", 
        min_value=5, 
        max_value=50, 
        value=12,
        help="Number of search results to display"
    )
    
    alpha = st.sidebar.slider(
        "Text vs Image Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.05,
        help="0.0 = image-only search, 1.0 = text-only search"
    )
    
    # Display alpha explanation
    if alpha > 0.8:
        st.sidebar.info("🔤 Heavily weighted toward text search")
    elif alpha > 0.6:
        st.sidebar.info("🔤 Text-focused search")
    elif alpha > 0.4:
        st.sidebar.info("⚖️ Balanced text/image search")
    elif alpha > 0.2:
        st.sidebar.info("🖼️ Image-focused search")
    else:
        st.sidebar.info("🖼️ Heavily weighted toward image search")
    
    # Actions
    st.sidebar.subheader("🔧 Actions")
    
    # Index folder button
    if st.sidebar.button("🗂️ Index Folder", type="primary"):
        if not screenshots_folder:
            st.sidebar.error("Please specify a screenshots folder")
        else:
            screenshots_path = Path(screenshots_folder)
            index_path = Path(index_folder)
            
            if not screenshots_path.exists():
                st.sidebar.error(f"Screenshots folder does not exist: {screenshots_folder}")
            elif not screenshots_path.is_dir():
                st.sidebar.error(f"Screenshots path is not a directory: {screenshots_folder}")
            else:
                # Clear any previous errors
                st.session_state.searcher_error = None
                
                # Build index with progress display
                build_index_with_progress(screenshots_path, index_path)
    
    # Offline mode checkbox
    offline_mode = st.sidebar.checkbox(
        "🔒 Offline Mode",
        value=os.environ.get("OFFLINE") == "1",
        help="Prevent any network requests (for privacy)"
    )
    
    # Set environment variable for offline mode
    if offline_mode:
        os.environ["OFFLINE"] = "1"
    else:
        os.environ.pop("OFFLINE", None)
    
    # Load index summary for sidebar display
    if Path(index_folder).exists():
        summary = load_index_summary(Path(index_folder))
        if summary:
            st.sidebar.subheader("📊 Index Status")
            st.sidebar.success("✅ Index Available")
            st.sidebar.write(f"**Images:** {summary['num_images']}")
            st.sidebar.write(f"**Built:** {format_timestamp(summary['built_at'])}")
            st.sidebar.write(f"**FAISS:** {'✅' if summary['have_faiss'] else '❌'}")
            
            if summary.get('processing_time'):
                st.sidebar.write(f"**Build Time:** {summary['processing_time']:.1f}s")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if index exists
        index_path = Path(index_folder)
        if not index_path.exists() or not (index_path / "meta.json").exists():
            st.warning("📂 No search index found. Please index a folder first using the sidebar.")
            
            if screenshots_folder:
                st.info(f"💡 Click '🗂️ Index Folder' in the sidebar to index: `{screenshots_folder}`")
            else:
                st.info("💡 First, specify a screenshots folder in the sidebar, then click '🗂️ Index Folder'")
            
            # Show some helpful info
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **Choose a folder** containing your screenshots in the sidebar
            2. **Click 'Index Folder'** to build the search index
            3. **Search** using natural language queries
            4. **Adjust the sliders** to fine-tune search behavior
            """)
            
            return
        
        # Load searcher
        searcher = load_searcher(index_folder)
        
        if searcher is None:
            st.error("❌ Failed to load search index")
            if hasattr(st.session_state, 'searcher_error') and st.session_state.searcher_error:
                st.error(f"Error: {st.session_state.searcher_error}")
            st.info("💡 Try rebuilding the index using the '🗂️ Index Folder' button in the sidebar")
            return
        
        # Search interface
        st.subheader("🔍 Search")
        
        query = st.text_input(
            "Search your screenshots",
            placeholder="e.g., 'receipt from coffee shop', 'boarding pass', 'error message'...",
            help="Use natural language to describe what you're looking for"
        )
        
        # Perform search if query provided
        if query.strip():
            try:
                with st.spinner("Searching..."):
                    results = searcher.search(
                        query=query.strip(),
                        top_k=top_k,
                        alpha=alpha
                    )
                
                if results:
                    st.success(f"🎯 Found {len(results)} results")
                    
                    # Display results in 3-column grid
                    cols = st.columns(3)
                    
                    for i, result in enumerate(results):
                        col_idx = i % 3
                        render_search_result(result, cols[col_idx])
                        
                else:
                    st.info("🤷‍♂️ No matching screenshots found")
                    st.markdown("**Try:**")
                    st.markdown("- Different keywords or phrases")
                    st.markdown("- Adjusting the text/image weight slider")
                    st.markdown("- Broader or more specific terms")
                    
            except Exception as e:
                st.error(f"Search failed: {e}")
        
        else:
            # Show example queries when no search is performed
            st.markdown("### 💡 Example Searches")
            
            examples = [
                "receipt total payment",
                "boarding pass flight gate",
                "error message dialog",
                "email notification",
                "code function python",
                "menu restaurant food",
                "invoice bill statement",
                "calendar meeting schedule"
            ]
            
            # Display examples in a nice grid
            example_cols = st.columns(2)
            for i, example in enumerate(examples):
                col = example_cols[i % 2]
                if col.button(f"🔍 {example}", key=f"example_{i}"):
                    st.rerun()
    
    with col2:
        # Right column for additional info/controls
        st.markdown("### ℹ️ Tips")
        st.markdown("""
        **Search Tips:**
        - Use descriptive keywords
        - Try partial phrases
        - Experiment with the sliders
        
        **Text Weight:**
        - High = finds text content
        - Low = finds visual similarity
        
        **Privacy:**
        - Everything runs locally
        - No data leaves your device
        - Enable offline mode for extra security
        """)


if __name__ == "__main__":
    main()