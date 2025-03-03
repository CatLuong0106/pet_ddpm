from pathlib import Path
from PIL import Image
import numpy as np

# Define the folder containing images
image_dir = Path("/home/luongcn/pet_ddpm/data/image_dir")  # Change this to your folder path

# Get all image files (filtering common image formats)
image_files = list(image_dir.glob("*.png"))  # Change to "*.jpg" or "*" for all files

# Iterate over images and print their shapes
for img_path in image_files:
    img = Image.open(img_path)  # Open image
    img_array = np.asarray(img)  # Convert to numpy array
    print(f"Image: {img_path.name}, Shape: {img_array.shape}")  # Print name and shape

