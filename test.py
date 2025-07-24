import cv2
import os

# Test with a simple path
simple_image_path = r"C:\Users\abhishik chebrolu\Downloads\AINN pro\UCSD\ucsdpeds\vidf\vidf1_33_000.y\vidf1_33_000_f002.png"  # Replace with a simple image path
print(f"Testing with path: {simple_image_path}")

# Check if the file exists
if os.path.isfile(simple_image_path):
    print(f"File exists: {simple_image_path}")
else:
    print(f"File does not exist: {simple_image_path}")

# Attempt to load the image
image = cv2.imread(simple_image_path)
if image is None:
    print("Failed to load image with OpenCV.")
else:
    print(f"Image loaded successfully, shape: {image.shape}")
