# Initialize with custom settings
processor = ImageProcessor(
    tile_size=(512, 512),
    max_workers=4,
    output_format=ImageFormat.PNG
)

# Process image with all options
tiles_info = processor.tile_image(
    input_path='large_image.tif',
    output_dir='output/tiles',
    preserve_alpha=True
)

# Reconstruct with custom format
transformed_labels = processor.reconstruct_image(
    tiles_dir='output/tiles',
    json_path='output/tiles/tiles_info.json',
    output_path='output/reconstructed.tiff',
    labels=your_labels,
    output_format=ImageFormat.TIFF
)