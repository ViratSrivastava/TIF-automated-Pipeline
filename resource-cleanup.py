import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Union
import threading
import time
import logging
from PIL import Image
import uuid

# Import the original ImageProcessor class
from image_processor import ImageProcessor, ImageFormat, ImageProcessingError

class CleanupManager:
    """Manages automatic cleanup of processed files"""
    
    def __init__(self, base_dir: Union[str, Path], retention_hours: float = 1.0):
        """
        Initialize cleanup manager
        
        Args:
            base_dir: Base directory for temporary files
            retention_hours: How long to keep files (default: 1 hour)
        """
        self.base_dir = Path(base_dir)
        self.retention_hours = retention_hours
        self.cleanup_thread = None
        self.should_run = False
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start the cleanup background thread"""
        self.should_run = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
    def stop(self):
        """Stop the cleanup background thread"""
        self.should_run = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
            
    def _cleanup_loop(self):
        """Background loop that periodically checks for old files"""
        while self.should_run:
            try:
                self._perform_cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")
            time.sleep(300)  # Check every 5 minutes
            
    def _perform_cleanup(self):
        """Remove old processed files"""
        if not self.base_dir.exists():
            return
            
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for item in self.base_dir.glob("*"):
            if item.is_dir():
                try:
                    # Check the directory's modified time
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff_time:
                        shutil.rmtree(item)
                        self.logger.info(f"Removed old directory: {item}")
                except Exception as e:
                    self.logger.error(f"Error removing directory {item}: {str(e)}")

class ProcessingSession:
    """Manages a single processing session with cleanup"""
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize processing session
        
        Args:
            base_dir: Base directory for all processing
        """
        self.base_dir = Path(base_dir)
        self.session_id = str(uuid.uuid4())
        self.session_dir = self.base_dir / self.session_id
        self.tiles_dir = self.session_dir / "tiles"
        self.output_dir = self.session_dir / "output"
        
        # Create directories
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = ImageProcessor()
        
    def process_image(self, 
                     input_path: Union[str, Path],
                     tile_size: tuple = (512, 512),
                     output_format: ImageFormat = ImageFormat.PNG,
                     labels: Optional[Dict] = None) -> Dict:
        """
        Process an image with automatic cleanup
        
        Args:
            input_path: Path to input image
            tile_size: Size of tiles
            output_format: Output image format
            labels: Optional labels for tiles
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            # Update processor configuration
            self.processor = ImageProcessor(
                tile_size=tile_size,
                output_format=output_format
            )
            
            # Generate tiles
            tiles_info = self.processor.tile_image(
                input_path=input_path,
                output_dir=self.tiles_dir,
                preserve_alpha=True
            )
            
            # Process with labels if provided
            if labels:
                output_path = self.output_dir / f"reconstructed.{output_format.name.lower()}"
                transformed_labels = self.processor.reconstruct_image(
                    tiles_dir=self.tiles_dir,
                    json_path=self.tiles_dir / "tiles_info.json",
                    output_path=output_path,
                    labels=labels,
                    output_format=output_format
                )
            else:
                output_path = self.output_dir / f"reconstructed.{output_format.name.lower()}"
                self.processor.reconstruct_image(
                    tiles_dir=self.tiles_dir,
                    json_path=self.tiles_dir / "tiles_info.json",
                    output_path=output_path,
                    output_format=output_format
                )
                transformed_labels = None
            
            return {
                "session_id": self.session_id,
                "tiles_dir": str(self.tiles_dir),
                "output_path": str(output_path),
                "tiles_info": tiles_info,
                "transformed_labels": transformed_labels
            }
            
        except Exception as e:
            raise ImageProcessingError(f"Processing session failed: {str(e)}")
            
    def cleanup(self):
        """Remove all files from this session"""
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
        except Exception as e:
            logging.error(f"Error cleaning up session {self.session_id}: {str(e)}")

# Global cleanup manager
cleanup_manager = None

def initialize_cleanup(base_dir: Union[str, Path], retention_hours: float = 1.0):
    """Initialize the global cleanup manager"""
    global cleanup_manager
    cleanup_manager = CleanupManager(base_dir, retention_hours)
    cleanup_manager.start()

def shutdown_cleanup():
    """Shutdown the global cleanup manager"""
    global cleanup_manager
    if cleanup_manager:
        cleanup_manager.stop()