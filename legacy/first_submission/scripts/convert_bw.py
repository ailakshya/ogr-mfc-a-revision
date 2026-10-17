import sys
import os
from PIL import Image, ImageEnhance

input_dir = "/Users/lakshya/.gemini/antigravity/brain/1e22ae0b-0ad1-4cb0-a02d-940034521864"
output_dir = input_dir

images = [
    "media__1775648796236.jpg",
    "media__1775648796265.jpg",
    "media__1775648796268.jpg",
    "media__1775648796294.jpg",
    "media__1775648796812.jpg"
]

for img_name in images:
    try:
        path = os.path.join(input_dir, img_name)
        img = Image.open(path)
        
        # Convert to grayscale
        gray = img.convert('L')
        
        # Enhance contrast slightly to look more professional/crisp
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.1)
        
        out_name = "exact_match_bw_" + img_name
        out_path = os.path.join(output_dir, out_name)
        gray.save(out_path)
        print("Saved", out_name)
    except Exception as e:
        print("Error on", img_name, e)
