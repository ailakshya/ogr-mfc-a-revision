import sys
import os
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

img_path = "/Users/lakshya/.gemini/antigravity/brain/1e22ae0b-0ad1-4cb0-a02d-940034521864/media__1775649344871.png"
out_path = "/Users/lakshya/.gemini/antigravity/brain/1e22ae0b-0ad1-4cb0-a02d-940034521864/native_perfect_face.png"

try:
    img = Image.open(img_path).convert("RGBA")
    
    # grayscale
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.2)
    gray = gray.convert("RGBA")
    
    width, height = gray.size
    
    # to soften the background, let's create a radial mask to whiten the edges
    # this acts like a poor man's background blur/removal making the subject pop
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # We estimate the face is in the top-center
    center_x = width // 2
    center_y = height // 3
    # Radius of the clear area
    r_x = int(width * 0.4)
    r_y = int(height * 0.5)
    
    draw.ellipse((center_x - r_x, center_y - r_y, center_x + r_x, center_y + r_y), fill=255)
    
    # blur the mask to make the transition smooth
    mask = mask.filter(ImageFilter.GaussianBlur(50))
    
    white_bg = Image.new("RGBA", (width, height), "WHITE")
    
    # Composite: where mask is 255 (center), we keep the gray image. Where 0 (edges), we get white.
    final = Image.composite(gray, white_bg, mask)
    
    final.save(out_path)
    print("Saved perfect native face.")
except Exception as e:
    print("Error:", e)
