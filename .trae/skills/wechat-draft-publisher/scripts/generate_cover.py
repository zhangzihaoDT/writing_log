from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_cover(output_path):
    # Dimensions: 1080x1440 (3:4 ratio) - though WeChat cover is usually 2.35:1 (900x383)
    # Wait, WeChat Draft Cover:
    # Large image: 2.35:1 (900x383)
    # Small image: 1:1 (200x200)
    # The user asked for "xhs-cover-designer" which generates 3:4 (1080x1440) for Little Red Book.
    # BUT, the goal is to publish to WeChat Draft.
    # WeChat official account cover usually needs to be landscape (2.35:1) for the main post.
    # However, if I use the XHS style (3:4), it will be cropped significantly in WeChat's timeline view.
    # Since the user explicitly asked to use "xhs-cover-designer", I should probably stick to that style 
    # OR adapt it. 
    # Let's generate a 1080x1440 image as requested by the skill, but put the main content in the center 
    # so it might survive some cropping, or just strictly follow the XHS skill spec.
    # Let's follow the XHS skill spec as requested: 1080x1440.
    
    width = 1080
    height = 1440
    
    # Colors (Tesla Vibe: Red & Black/Dark Grey)
    bg_color = (20, 20, 20) # Dark Grey
    text_color = (255, 255, 255) # White
    accent_color = (232, 33, 39) # Tesla Red
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Fonts
    # Try to find a Chinese font on macOS
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc"
    ]
    
    font_path = None
    for p in font_paths:
        if os.path.exists(p):
            font_path = p
            break
            
    if not font_path:
        print("Warning: No Chinese font found. Text might not render correctly.")
        # Fallback to default (might not support Chinese)
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    else:
        try:
            # PingFang.ttc often has multiple faces, index 0 is usually valid
            font_large = ImageFont.truetype(font_path, 120, index=0)
            font_medium = ImageFont.truetype(font_path, 60, index=0)
            font_small = ImageFont.truetype(font_path, 40, index=0)
        except Exception as e:
            print(f"Error loading font: {e}")
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Draw Decoration (Red shape)
    draw.rectangle([(0, 0), (width, 200)], fill=accent_color)
    
    # Title
    title = "特斯拉 7 年低息"
    # Calculate text position (Centered)
    # getbbox returns (left, top, right, bottom)
    bbox = font_large.getbbox(title)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) / 2
    y = (height - text_height) / 2 - 100
    draw.text((x, y), title, font=font_large, fill=text_color)
    
    # Subtitle
    subtitle = "中产的时间游戏"
    bbox_sub = font_medium.getbbox(subtitle)
    text_width_sub = bbox_sub[2] - bbox_sub[0]
    
    x_sub = (width - text_width_sub) / 2
    y_sub = y + text_height + 40
    draw.text((x_sub, y_sub), subtitle, font=font_medium, fill=text_color)
    
    # Footer/CTA
    cta = "深度好文"
    bbox_cta = font_small.getbbox(cta)
    text_width_cta = bbox_cta[2] - bbox_cta[0]
    
    x_cta = (width - text_width_cta) / 2
    y_cta = height - 200
    
    # Draw CTA background pill
    padding = 20
    draw.rounded_rectangle(
        [x_cta - padding, y_cta - padding, x_cta + text_width_cta + padding, y_cta + bbox_cta[3] + padding],
        radius=10,
        fill=accent_color
    )
    draw.text((x_cta, y_cta), cta, font=font_small, fill=text_color)

    # Save
    img.save(output_path)
    print(f"Cover image generated at: {output_path}")

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "cover.png"
    create_cover(output_file)
