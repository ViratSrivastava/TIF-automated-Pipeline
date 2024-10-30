import streamlit as st
import os
from pathlib import Path
import json
import time
from PIL import Image
import io
import base64
from typing import Optional

from image_processor import ImageFormat, ImageProcessingError
from cleanup_processor import ProcessingSession, initialize_cleanup, shutdown_cleanup

# Configure the application
st.set_page_config(
    page_title="Image Processing Pipeline",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate download link for binary file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("Image Processing Pipeline")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    tile_size = st.sidebar.number_input(
        "Tile Size",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Size of image tiles in pixels"
    )
    
    output_format = st.sidebar.selectbox(
        "Output Format",
        options=[format.name for format in ImageFormat],
        index=0,
        help="Format for output images"
    )
    
    retention_hours = st.sidebar.number_input(
        "File Retention (hours)",
        min_value=0.1,
        max_value=24.0,
        value=1.0,
        step=0.1,
        help="How long to keep processed files"
    )
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        # Display original image
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process button
        if st.button("Process Image"):
            try:
                # Create temporary directory for this upload
                base_dir = Path("temp_processing")
                base_dir.mkdir(exist_ok=True)
                
                # Initialize cleanup
                initialize_cleanup(base_dir, retention_hours)
                
                # Create processing session
                session = ProcessingSession(base_dir)
                
                # Save uploaded file
                temp_input = session.session_dir / uploaded_file.name
                with open(temp_input, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Process image
                with st.spinner("Processing image..."):
                    result = session.process_image(
                        input_path=temp_input,
                        tile_size=(tile_size, tile_size),
                        output_format=ImageFormat[output_format]
                    )
                
                # Display results
                st.success("Processing complete!")
                
                # Show reconstructed image
                st.subheader("Reconstructed Image")
                reconstructed = Image.open(result["output_path"])
                st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)
                
                # Download links
                st.subheader("Downloads")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        get_binary_file_downloader_html(
                            result["output_path"],
                            "Reconstructed Image"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Save tiles info to downloadable JSON
                    tiles_info_path = Path(result["tiles_dir"]) / "tiles_info.json"
                    if tiles_info_path.exists():
                        st.markdown(
                            get_binary_file_downloader_html(
                                tiles_info_path,
                                "Tiles Information"
                            ),
                            unsafe_allow_html=True
                        )
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": uploaded_file.name,
                    "session_id": result["session_id"]
                })
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Display processing history
    if st.session_state.processing_history:
        st.subheader("Processing History")
        history_df = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(history_df)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure cleanup manager is properly shutdown
        shutdown_cleanup()