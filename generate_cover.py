from PIL import Image, ImageDraw, ImageFont
import os

def create_cover(title, output_path="cover.png"):
    width, height = 900, 383
    # Hara style: White background
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf"
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                # Use a large font size
                font = ImageFont.truetype(path, 60)
                break
            except Exception:
                continue
                
    if font is None:
        font = ImageFont.load_default()

    # Draw title centered
    # Get bounding box
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    
    # Hara style: Black text
    draw.text((x, y), title, font=font, fill='black')
    
    # Add a subtitle or "Hara Editor" mark
    try:
        small_font = ImageFont.truetype(path, 24)
    except:
        small_font = font
        
    mark = "WeChat Hara Editor"
    bbox_mark = draw.textbbox((0, 0), mark, font=small_font)
    mark_width = bbox_mark[2] - bbox_mark[0]
    
    draw.text(((width - mark_width) / 2, y + text_height + 20), mark, font=small_font, fill='gray')

    image.save(output_path)
    print(f"Cover image saved to {output_path}")

if __name__ == "__main__":
    create_cover("漫长的季节")
