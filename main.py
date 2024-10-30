import os
import json
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import multiprocessing
from enum import Enum
import warnings
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFormat(Enum):
    """Supported image formats"""
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    BMP = "BMP"
    WEBP = "WEBP"

@dataclass
class Coordinates:
    """Dataclass for storing coordinates"""
    x: int
    y: int
    
    def __post_init__(self):
        if not isinstance(self.x, (int, float)) or not isinstance(self.y, (int, float)):
            raise ValueError("Coordinates must be numeric")
        
@dataclass
class BoundingBox:
    """Dataclass for storing bounding box coordinates"""
    left: int
    upper: int
    right: int
    lower: int
    
    def __post_init__(self):
        if self.left > self.right or self.upper > self.lower:
            raise ValueError("Invalid bounding box coordinates")
            
@dataclass
class ImageSize:
    """Dataclass for storing image dimensions"""
    width: int
    height: int

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

def process_tile(args: tuple) -> dict:
    """
    Process a single tile in parallel execution
    
    Args:
        args: Tuple containing (image_path, tile_bbox, output_path, tile_filename)
        
    Returns:
        Dictionary containing tile information
    """
    try:
        image_path, bbox, output_path, tile_filename = args
        with Image.open(image_path) as img:
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            tile = img.crop((bbox.left, bbox.upper, bbox.right, bbox.lower))
            tile_path = os.path.join(output_path, tile_filename)
            tile.save(tile_path)
            
            return {
                'filename': tile_filename,
                'bbox': asdict(bbox),
                'size': asdict(ImageSize(tile.width, tile.height))
            }
    except Exception as e:
        logger.error(f"Error processing tile {tile_filename}: {str(e)}")
        raise ImageProcessingError(f"Failed to process tile: {str(e)}")

class ImageProcessor:
    def __init__(self, 
                 tile_size: Tuple[int, int] = (256, 256),
                 max_workers: Optional[int] = None,
                 output_format: ImageFormat = ImageFormat.PNG):
        """
        Initialize the image processor with configuration
        
        Args:
            tile_size: Tuple of (width, height) for each tile
            max_workers: Maximum number of parallel workers (default: CPU count)
            output_format: Output format for tiles (default: PNG)
        """
        self.validate_tile_size(tile_size)
        self.tile_size = tile_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.output_format = output_format
        
    @staticmethod
    def validate_tile_size(tile_size: Tuple[int, int]):
        """Validate tile size parameters"""
        if not isinstance(tile_size, tuple) or len(tile_size) != 2:
            raise ValueError("tile_size must be a tuple of (width, height)")
        if not all(isinstance(x, int) and x > 0 for x in tile_size):
            raise ValueError("tile_size dimensions must be positive integers")

    def validate_image(self, image_path: Union[str, Path]) -> None:
        """
        Validate input image file
        
        Args:
            image_path: Path to input image
            
        Raises:
            ImageProcessingError: If image is invalid
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ImageProcessingError(f"Invalid image file: {str(e)}")

    def tile_image(self, 
                  input_path: Union[str, Path], 
                  output_dir: Union[str, Path],
                  preserve_alpha: bool = False) -> Dict:
        """
        Break an image into tiles using parallel processing
        
        Args:
            input_path: Path to input image
            output_dir: Directory to save tiles and JSON
            preserve_alpha: Whether to preserve alpha channel
            
        Returns:
            Dictionary containing tile information
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # Validate inputs
        self.validate_image(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process image
        try:
            with Image.open(input_path) as img:
                # Handle color mode
                if preserve_alpha and img.mode == 'RGBA':
                    mode = 'RGBA'
                else:
                    mode = 'RGB'
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Calculate tiles
                tiles_info = {
                    'original_size': asdict(ImageSize(width, height)),
                    'tile_size': asdict(ImageSize(*self.tile_size)),
                    'color_mode': mode,
                    'tiles': []
                }
                
                # Prepare parallel processing tasks
                tasks = []
                for i in range(0, height, self.tile_size[1]):
                    for j in range(0, width, self.tile_size[0]):
                        bbox = BoundingBox(
                            left=j,
                            upper=i,
                            right=min(j + self.tile_size[0], width),
                            lower=min(i + self.tile_size[1], height)
                        )
                        
                        tile_filename = f'tile_{i}_{j}.{self.output_format.name.lower()}'
                        tasks.append((
                            input_path,
                            bbox,
                            output_dir,
                            tile_filename
                        ))
                
                # Process tiles in parallel
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_tile = {executor.submit(process_tile, task): task 
                                    for task in tasks}
                    
                    for future in as_completed(future_to_tile):
                        try:
                            tile_info = future.result()
                            tiles_info['tiles'].append(tile_info)
                        except Exception as e:
                            logger.error(f"Tile processing failed: {str(e)}")
                            raise
                
                # Save tiles information
                tiles_info['tiles'].sort(key=lambda x: x['filename'])
                json_path = output_dir / 'tiles_info.json'
                with open(json_path, 'w') as f:
                    json.dump(tiles_info, f, indent=2)
                
                return tiles_info
                
        except Exception as e:
            raise ImageProcessingError(f"Failed to process image: {str(e)}")

    def reconstruct_image(self, 
                         tiles_dir: Union[str, Path],
                         json_path: Union[str, Path],
                         output_path: Union[str, Path],
                         labels: Optional[Dict[str, List]] = None,
                         output_format: Optional[ImageFormat] = None) -> Dict[str, List]:
        """
        Reconstruct image from tiles and transform labels
        
        Args:
            tiles_dir: Directory containing tiles
            json_path: Path to tiles information JSON
            output_path: Path to save reconstructed image
            labels: Dictionary of labels for each tile
            output_format: Output format for final image
            
        Returns:
            Dictionary of transformed labels
        """
        tiles_dir = Path(tiles_dir)
        json_path = Path(json_path)
        output_path = Path(output_path)
        
        try:
            # Load tiles information
            with open(json_path, 'r') as f:
                tiles_info = json.load(f)
            
            # Create blank image
            width = tiles_info['original_size']['width']
            height = tiles_info['original_size']['height']
            mode = tiles_info.get('color_mode', 'RGB')
            reconstructed = Image.new(mode, (width, height))
            
            transformed_labels = {}
            
            # Process tiles in parallel
            def process_reconstruction_tile(tile_info):
                tile_path = tiles_dir / tile_info['filename']
                with Image.open(tile_path) as tile:
                    bbox = BoundingBox(**tile_info['bbox'])
                    return {
                        'tile': tile.copy(),
                        'bbox': bbox,
                        'filename': tile_info['filename']
                    }
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_tile = {executor.submit(process_reconstruction_tile, tile_info): tile_info 
                                for tile_info in tiles_info['tiles']}
                
                for future in as_completed(future_to_tile):
                    try:
                        result = future.result()
                        bbox = result['bbox']
                        reconstructed.paste(result['tile'], (bbox.left, bbox.upper))
                        
                        # Transform labels if provided
                        if labels and result['filename'] in labels:
                            tile_labels = labels[result['filename']]
                            transformed = []
                            for label in tile_labels:
                                if isinstance(label, dict) and 'coordinates' in label:
                                    new_label = label.copy()
                                    new_coords = Coordinates(
                                        x=label['coordinates']['x'] + bbox.left,
                                        y=label['coordinates']['y'] + bbox.upper
                                    )
                                    new_label['coordinates'] = asdict(new_coords)
                                    transformed.append(new_label)
                            transformed_labels[result['filename']] = transformed
                    except Exception as e:
                        logger.error(f"Tile reconstruction failed: {str(e)}")
                        raise
            
            # Save reconstructed image
            output_format = output_format or ImageFormat.TIFF
            save_kwargs = {}
            
            if output_format == ImageFormat.TIFF:
                save_kwargs['compression'] = 'tiff_lzw'
            elif output_format == ImageFormat.JPEG:
                save_kwargs['quality'] = 95
            elif output_format == ImageFormat.WEBP:
                save_kwargs['quality'] = 90
                save_kwargs['method'] = 6
            
            reconstructed.save(output_path, format=output_format.value, **save_kwargs)
            
            return transformed_labels
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to reconstruct image: {str(e)}")

def main():
    """Example usage of the enhanced pipeline"""
    try:
        # Initialize processor
        processor = ImageProcessor(
            tile_size=(512, 512),
            max_workers=4,
            output_format=ImageFormat.PNG
        )
        
        # Example tiling
        tiles_info = processor.tile_image(
            input_path='input.tif',
            output_dir='tiles',
            preserve_alpha=True
        )
        
        # Example labels
        example_labels = {
            'tile_0_0.png': [
                {
                    'label': 'object1',
                    'coordinates': {'x': 100, 'y': 100},
                    'confidence': 0.95
                }
            ]
        }
        
        # Example reconstruction
        transformed_labels = processor.reconstruct_image(
            tiles_dir='tiles',
            json_path='tiles/tiles_info.json',
            output_path='reconstructed.tif',
            labels=example_labels,
            output_format=ImageFormat.TIFF
        )
        
    except ImageProcessingError as e:
        logger.error(f"Processing failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    main()

