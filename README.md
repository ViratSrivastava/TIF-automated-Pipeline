
# TIF-Automated-Pipeline Image Processing Pipeline

A robust Python pipeline for processing large images through tiling, parallel processing, and reconstruction. This tool is designed to handle various image formats, maintain spatial coordinates, and process labels across transformations.

## Features

- **Parallel Processing**: Efficient handling of large images using multi-core processing
- **Multiple Format Support**: Handles TIFF, PNG, JPEG, BMP, and WebP formats
- **Coordinate Tracking**: Maintains precise spatial information across transformations
- **Label Management**: Transforms labels from tile coordinates to original image coordinates
- **Error Handling**: Comprehensive validation and error recovery mechanisms
- **Memory Efficient**: Processes large images in chunks to minimize memory usage

## Installation

### Prerequisites

- Python 3.7 or higher
- Pillow (PIL) library
- concurrent.futures (included in Python standard library)

```bash
pip install Pillow
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/image-processing-pipeline.git
cd image-processing-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from image_processor import ImageProcessor, ImageFormat

# Initialize processor
processor = ImageProcessor(
    tile_size=(512, 512),  # Size of each tile
    max_workers=4,         # Number of parallel workers
    output_format=ImageFormat.PNG  # Output format for tiles
)

# Break image into tiles
tiles_info = processor.tile_image(
    input_path='input.tif',
    output_dir='output/tiles',
    preserve_alpha=True
)

# Process tiles and generate labels (your custom processing here)
labels = {
    'tile_0_0.png': [
        {
            'label': 'object1',
            'coordinates': {'x': 100, 'y': 100},
            'confidence': 0.95
        }
    ]
}

# Reconstruct image and transform labels
transformed_labels = processor.reconstruct_image(
    tiles_dir='output/tiles',
    json_path='output/tiles/tiles_info.json',
    output_path='output/reconstructed.tiff',
    labels=labels,
    output_format=ImageFormat.TIFF
)
```

### Advanced Configuration

#### Custom Tile Sizes
```python
processor = ImageProcessor(
    tile_size=(1024, 1024),  # Larger tiles
    max_workers=8            # More parallel workers
)
```

#### Format-Specific Settings
```python
# JPEG output with quality settings
processor = ImageProcessor(output_format=ImageFormat.JPEG)
transformed_labels = processor.reconstruct_image(
    # ... other parameters ...
    output_format=ImageFormat.JPEG
)

# TIFF output with compression
processor = ImageProcessor(output_format=ImageFormat.TIFF)
transformed_labels = processor.reconstruct_image(
    # ... other parameters ...
    output_format=ImageFormat.TIFF
)
```

## Directory Structure

```
image-processing-pipeline/
├── image_processor.py     # Main processing code
├── requirements.txt       # Python dependencies
├── examples/             # Example scripts and notebooks
├── tests/               # Unit tests
└── output/              # Default output directory
    ├── tiles/          # Generated tiles
    └── reconstructed/  # Reconstructed images
```

## Output Format

### Tile Information JSON
```json
{
    "original_size": {
        "width": 5000,
        "height": 4000
    },
    "tile_size": {
        "width": 512,
        "height": 512
    },
    "color_mode": "RGB",
    "tiles": [
        {
            "filename": "tile_0_0.png",
            "bbox": {
                "left": 0,
                "upper": 0,
                "right": 512,
                "lower": 512
            },
            "size": {
                "width": 512,
                "height": 512
            }
        }
    ]
}
```

### Label Format
```json
{
    "tile_0_0.png": [
        {
            "label": "object1",
            "coordinates": {
                "x": 100,
                "y": 100
            },
            "confidence": 0.95
        }
    ]
}
```

## Error Handling

The pipeline includes comprehensive error handling:

```python
from image_processor import ImageProcessor, ImageProcessingError

try:
    processor = ImageProcessor()
    tiles_info = processor.tile_image(...)
except ImageProcessingError as e:
    print(f"Processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Tile Size**: Choose tile sizes based on your image and memory constraints
   - Larger tiles = fewer I/O operations but more memory usage
   - Smaller tiles = more I/O operations but less memory usage

2. **Parallel Processing**: Adjust `max_workers` based on your system
   - CPU-bound: Use `multiprocessing.cpu_count()`
   - I/O-bound: Can use more workers than CPU cores

3. **Format Selection**: Choose formats based on your needs
   - TIFF: Lossless, good for scientific data
   - JPEG: Smaller size, good for photos
   - PNG: Lossless, good for images with text/lines
   - WebP: Modern format with good compression

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce tile size
   - Reduce number of parallel workers
   - Check available system memory

2. **Slow Processing**
   - Increase tile size
   - Adjust number of workers
   - Check disk I/O performance

3. **Image Quality Issues**
   - Check input image format and quality
   - Adjust format-specific settings
   - Verify color mode handling

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/your-repo/image-processing-pipeline](https://github.com/your-repo/image-processing-pipeline)

## Acknowledgments

- Pillow (PIL) library
- Python multiprocessing
- Your additional acknowledgments