#!/usr/bin/env python3
import os
import urllib.request
import ssl

# Disable SSL verification for simplicity
ssl._create_default_https_context = ssl._create_unverified_context

def download_sample_images():
    """Download sample images from USC SIPI database"""
    
    os.makedirs('input_images', exist_ok=True)
    
    # USC SIPI Image Database URLs
    base_url = "https://sipi.usc.edu/database/download.php?vol=misc&img="
    
    images = [
        "4.2.03",  # Jelly beans
        "4.2.04",  # Jelly beans 2
        "4.2.05",  # House
        "4.2.06",  # Clock
        "4.2.07",  # Butterfly
        "5.1.09",  # Tiffany
        "5.1.10",  # Tiffany
        "5.1.11",  # Tiffany
        "5.1.12",  # Stream
        "5.1.13",  # Stream bridge
    ]
    
    print("Downloading sample images...")
    
    for i, img_id in enumerate(images):
        # Download multiple copies to get to 200+ images
        for copy in range(25):  # 10 images x 25 copies = 250 images
            url = base_url + img_id
            filename = f"input_images/image_{i:03d}_{copy:03d}.tiff"
            
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(f"Failed to download {img_id}: {e}")
                continue
    
    print(f"\nDownloaded images to input_images/")
    print(f"Total images: {len(os.listdir('input_images'))}")

if __name__ == "__main__":
    download_sample_images()