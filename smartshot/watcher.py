"""
File watcher for Smart Screenshot Search.

Monitors a folder for new screenshot files and automatically rebuilds
the search index when changes are detected.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Event, Timer
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from watchdog.observers import Observer

from .indexer import build_index
from .utils import IMG_EXTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ScreenshotWatcher(FileSystemEventHandler):
    """
    File system event handler for screenshot files.
    
    Monitors for new image files and triggers index rebuilding with debouncing
    to avoid multiple rebuilds for rapid file changes.
    """
    
    def __init__(self, screenshots_dir: Path, index_dir: Path, debounce_seconds: float = 3.0):
        """
        Initialize the screenshot watcher.
        
        Args:
            screenshots_dir: Directory to monitor for new screenshots
            index_dir: Directory where index files are stored
            debounce_seconds: Delay before triggering rebuild after last change
        """
        super().__init__()
        self.screenshots_dir = Path(screenshots_dir)
        self.index_dir = Path(index_dir)
        self.debounce_seconds = debounce_seconds
        
        # Debouncing timer
        self._rebuild_timer: Optional[Timer] = None
        self._pending_changes = set()
        
        # Track when we're rebuilding to avoid concurrent rebuilds
        self._rebuilding = False
        
        logger.info(f"Watching {self.screenshots_dir} for new screenshots")
        logger.info(f"Index directory: {self.index_dir}")
        logger.info(f"Debounce delay: {self.debounce_seconds}s")
    
    def _is_image_file(self, file_path: Path) -> bool:
        """
        Check if a file is a supported image format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a supported image format
        """
        return file_path.suffix.lower() in IMG_EXTS
    
    def _schedule_rebuild(self, file_path: Path) -> None:
        """
        Schedule an index rebuild with debouncing.
        
        Args:
            file_path: Path of the file that triggered the change
        """
        if self._rebuilding:
            logger.debug(f"Rebuild already in progress, ignoring {file_path.name}")
            return
        
        # Add to pending changes
        self._pending_changes.add(file_path)
        
        # Cancel existing timer
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
        
        # Schedule new timer
        self._rebuild_timer = Timer(self.debounce_seconds, self._execute_rebuild)
        self._rebuild_timer.start()
        
        logger.info(f"Scheduled rebuild for {file_path.name} (debounce: {self.debounce_seconds}s)")
    
    def _execute_rebuild(self) -> None:
        """
        Execute the index rebuild.
        """
        if self._rebuilding:
            logger.warning("Rebuild already in progress, skipping")
            return
        
        try:
            self._rebuilding = True
            
            # Get list of pending changes
            pending_files = list(self._pending_changes)
            self._pending_changes.clear()
            
            if not pending_files:
                logger.debug("No pending changes, skipping rebuild")
                return
            
            logger.info(f"Starting index rebuild for {len(pending_files)} file(s)")
            for file_path in pending_files:
                logger.info(f"  - {file_path.name}")
            
            # Rebuild the index
            start_time = time.time()
            summary = build_index(self.screenshots_dir, self.index_dir)
            elapsed = time.time() - start_time
            
            logger.info(f"✓ Index rebuild completed in {elapsed:.1f}s")
            logger.info(f"  Indexed {summary['num_images']} images")
            logger.info(f"  FAISS: {summary.get('have_faiss', False)}")
            logger.info(f"  NumPy: {summary.get('have_numpy', False)}")
            
        except Exception as e:
            logger.error(f"✗ Index rebuild failed: {e}")
            
        finally:
            self._rebuilding = False
    
    def on_created(self, event):
        """
        Handle file creation events.
        
        Args:
            event: File system event
        """
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            file_path = Path(event.src_path)
            
            if self._is_image_file(file_path):
                logger.info(f"New image detected: {file_path.name}")
                self._schedule_rebuild(file_path)
            else:
                logger.debug(f"Ignoring non-image file: {file_path.name}")
    
    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: File system event
        """
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            file_path = Path(event.src_path)
            
            if self._is_image_file(file_path):
                logger.info(f"Image modified: {file_path.name}")
                self._schedule_rebuild(file_path)
    
    def cleanup(self) -> None:
        """
        Clean up resources and cancel pending operations.
        """
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
            logger.info("Cancelled pending rebuild timer")
        
        if self._rebuilding:
            logger.info("Waiting for current rebuild to complete...")
            # Note: We can't really stop a rebuild in progress,
            # but we can at least log that we're waiting


class WatcherService:
    """
    Main watcher service that coordinates the file observer and event handler.
    """
    
    def __init__(self, screenshots_dir: Path, index_dir: Path, debounce_seconds: float = 3.0):
        """
        Initialize the watcher service.
        
        Args:
            screenshots_dir: Directory to monitor
            index_dir: Directory for index files
            debounce_seconds: Debounce delay for rebuilds
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.index_dir = Path(index_dir)
        self.debounce_seconds = debounce_seconds
        
        # Create directories if they don't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.event_handler = ScreenshotWatcher(
            self.screenshots_dir, 
            self.index_dir, 
            self.debounce_seconds
        )
        self.observer = Observer()
        self.stop_event = Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_names = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}
        signal_name = signal_names.get(signum, f'Signal {signum}')
        
        logger.info(f"Received {signal_name}, shutting down gracefully...")
        self.stop()
    
    def start(self) -> None:
        """
        Start the file watcher.
        
        Raises:
            FileNotFoundError: If screenshots directory doesn't exist
        """
        # Validate screenshots directory
        if not self.screenshots_dir.exists():
            raise FileNotFoundError(f"Screenshots directory not found: {self.screenshots_dir}")
        
        if not self.screenshots_dir.is_dir():
            raise NotADirectoryError(f"Screenshots path is not a directory: {self.screenshots_dir}")
        
        # Start watching
        self.observer.schedule(
            self.event_handler,
            str(self.screenshots_dir),
            recursive=False  # Only watch the specified directory, not subdirectories
        )
        
        self.observer.start()
        
        logger.info("=== Smart Screenshot Search Watcher Started ===")
        logger.info(f"Monitoring: {self.screenshots_dir}")
        logger.info(f"Index directory: {self.index_dir}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Main loop - wait for stop signal
            while not self.stop_event.is_set():
                time.sleep(1)
                
        except KeyboardInterrupt:
            # This shouldn't normally be reached due to signal handler,
            # but included as a backup
            logger.info("Keyboard interrupt received")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Stop the file watcher and clean up resources.
        """
        if self.stop_event.is_set():
            return  # Already stopping
        
        logger.info("Stopping watcher...")
        self.stop_event.set()
        
        # Stop the observer
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5.0)  # Wait up to 5 seconds
            
            if self.observer.is_alive():
                logger.warning("Observer did not stop gracefully")
            else:
                logger.info("Observer stopped")
        
        # Clean up event handler
        self.event_handler.cleanup()
        
        logger.info("Watcher stopped")


def main():
    """
    CLI entry point for the screenshot watcher.
    """
    parser = argparse.ArgumentParser(
        description='Watch folder for new screenshots and rebuild search index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--folder', '--screenshots', 
        required=True, 
        type=Path,
        help='Directory to monitor for new screenshots'
    )
    
    parser.add_argument(
        '--index', '--output',
        required=True,
        type=Path, 
        help='Directory to store/update index files'
    )
    
    parser.add_argument(
        '--debounce',
        type=float,
        default=3.0,
        help='Seconds to wait before rebuilding after last file change'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Create and start watcher service
        watcher = WatcherService(
            screenshots_dir=args.folder,
            index_dir=args.index,
            debounce_seconds=args.debounce
        )
        
        watcher.start()
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        return 1
    except NotADirectoryError as e:
        logger.error(f"Invalid directory: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Watcher failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())