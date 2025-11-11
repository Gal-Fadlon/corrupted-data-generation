#!/usr/bin/env python3
"""
Create a dummy dataset for testing Ambient Diffusion
Generates random colored images and saves them in a directory
"""

import os
import numpy as np
from PIL import Image
import argparse

def create_dummy_dataset(output_dir, num_images=100, image_size=32, num_classes=10):
    """
    Create a dummy dataset with random images
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of square images (e.g., 32 for 32x32)
        num_classes: Number of classes (for optional labels)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} dummy images of size {image_size}x{image_size}...")
    
    # Create images
    for i in range(num_images):
        # Generate random RGB image
        # Use different patterns for variety
        if i % 4 == 0:
            # Random noise
            img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        elif i % 4 == 1:
            # Solid colors
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            img_array = np.full((image_size, image_size, 3), color, dtype=np.uint8)
        elif i % 4 == 2:
            # Gradient
            img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            for x in range(image_size):
                for y in range(image_size):
                    img_array[x, y] = [
                        int(255 * x / image_size),
                        int(255 * y / image_size),
                        int(255 * (x + y) / (2 * image_size))
                    ]
        else:
            # Checkerboard pattern
            img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            block_size = max(1, image_size // 8)
            for x in range(image_size):
                for y in range(image_size):
                    if ((x // block_size) + (y // block_size)) % 2 == 0:
                        img_array[x, y] = [255, 255, 255]
                    else:
                        img_array[x, y] = np.random.randint(0, 256, 3)
        
        # Create PIL Image and save
        img = Image.fromarray(img_array, 'RGB')
        
        # Save with zero-padded filename
        filename = f"image_{i:05d}.png"
        img.save(os.path.join(output_dir, filename))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_images} images...")
    
    print(f"\n✓ Dataset created successfully in: {output_dir}")
    print(f"  Total images: {num_images}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"\nYou can now use: --data={output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy dataset for Ambient Diffusion")
    parser.add_argument("--output", type=str, default="./data/dummy_cifar10", 
                        help="Output directory for images")
    parser.add_argument("--num_images", type=int, default=100, 
                        help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=32, 
                        help="Size of square images")
    
    args = parser.parse_args()
    
    create_dummy_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=args.image_size
    )

