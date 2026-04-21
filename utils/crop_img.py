"""
Simplified crop_img.py
✔ No dlib
✔ No CUDA extensions
✔ No MTCNN
✔ Works on CPU
✔ Safe fallback for Talk-to-Edit
"""

import os
import cv2
from PIL import Image


def crop_img(img_size, input_img_path, cropped_output_path, device='cuda'):
    """
    Safe fallback cropping:
    - Just resizes image to required size
    - No face detection
    - No external heavy dependencies
    """

    if not os.path.exists(input_img_path):
        raise FileNotFoundError(f"Input image not found: {input_img_path}")

    # Load image
    img = Image.open(input_img_path).convert("RGB")

    # Resize to required resolution
    img = img.resize((img_size, img_size), Image.BICUBIC)

    # Save output
    img.save(cropped_output_path)

    return True