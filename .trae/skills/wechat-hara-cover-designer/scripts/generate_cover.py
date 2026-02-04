import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import os
import random
import math

def create_hara_cover(title, subtitle=None, theme='light', output_path='cover.png', layout_mode='random'):
    # Specs: 2.35:1
    WIDTH = 2350
    HEIGHT = 1000
    
    # Theme Configuration
    THEMES = {
        'light': {
            'bg': '#FFFFFF',
            'text': '#000000',
            'accent': '#A61C00', # MUJI Red
            'grid': '#E0E0E0'
        },
        'dark': {
            'bg': '#1A1A1A',
            'text': '#FFFFFF',
            'accent': '#FFFFFF',
            'grid': '#333333'
        },
        'grey': {
            'bg': '#F2F2F2',
            'text': '#333333',
            'accent': '#000000',
            'grid': '#D0D0D0'
        }
    }
    
    colors = THEMES.get(theme, THEMES['light'])
    
    # Create Image
    img = Image.new('RGB', (WIDTH, HEIGHT), color=colors['bg'])
    draw = ImageDraw.Draw(img)
    
    # --- Fonts Strategy ---
    def load_font(size, font_candidates):
        for path, index in font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size, index=index)
                except Exception as e:
                    continue
        return ImageFont.load_default()

    local_font = os.path.join(os.path.dirname(__file__), "../assets/PingFang.ttc")
    
    font_candidates_title = [
        (local_font, 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0)
    ]
    
    font_candidates_subtitle = [
        (local_font, 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0)
    ]

    # --- Helper Functions ---
    def get_text_size(text, font):
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        except AttributeError:
            return draw.textsize(text, font=font)

    def add_noise(image, intensity=10):
        # Add subtle grain for "materiality"
        pixels = image.load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                if random.random() > 0.8: # Only affect 20% of pixels for performance
                    r, g, b = pixels[i, j]
                    noise = random.randint(-intensity, intensity)
                    pixels[i, j] = (
                        max(0, min(255, r + noise)),
                        max(0, min(255, g + noise)),
                        max(0, min(255, b + noise))
                    )

    def draw_grid(step=100):
        # Subtle grid for structure
        for x in range(0, WIDTH, step):
            draw.line([(x, 0), (x, HEIGHT)], fill=colors['grid'], width=1)
        for y in range(0, HEIGHT, step):
            draw.line([(0, y), (WIDTH, y)], fill=colors['grid'], width=1)

    # --- Layout Strategies ---

    def layout_horizon():
        # "The Horizon": Strong horizontal line, title above, subtitle below far right
        draw_grid(step=200) # Wide grid
        
        # Horizon Line
        horizon_y = int(HEIGHT * 0.618) # Golden ratio
        draw.line([(0, horizon_y), (WIDTH, horizon_y)], fill=colors['text'], width=4)
        
        # Title (Large, Top Left)
        t_size = 180
        t_font = load_font(t_size, font_candidates_title)
        draw.text((150, horizon_y - 250), title, font=t_font, fill=colors['text'])
        
        # Subtitle (Small, Bottom Right)
        if subtitle:
            s_size = 60
            s_font = load_font(s_size, font_candidates_subtitle)
            sw, sh = get_text_size(subtitle, s_font)
            draw.text((WIDTH - sw - 150, horizon_y + 60), subtitle, font=s_font, fill=colors['text'])
            
            # Accent
            draw.rectangle([WIDTH - sw - 150, horizon_y - 10, WIDTH - sw - 150 + 40, horizon_y], fill=colors['accent'])

    def layout_void():
        # "The Void": Massive whitespace, small centered content or corner anchor
        # Title Center
        t_size = 120
        t_font = load_font(t_size, font_candidates_title)
        tw, th = get_text_size(title, t_font)
        
        cx, cy = WIDTH // 2, HEIGHT // 2
        draw.text((cx - tw // 2, cy - th // 2 - 40), title, font=t_font, fill=colors['text'])
        
        if subtitle:
            s_size = 40
            s_font = load_font(s_size, font_candidates_subtitle)
            sw, sh = get_text_size(subtitle, s_font)
            draw.text((cx - sw // 2, cy + th // 2 + 20), subtitle, font=s_font, fill=colors['text'])
        
        # Geometric Anchor (Corner)
        # Draw a circle in bottom right
        r = 150
        draw.ellipse([WIDTH - 100 - r, HEIGHT - 100 - r, WIDTH - 100, HEIGHT - 100], outline=colors['accent'], width=8)
        # Draw a line from circle to text? No, too messy. Keep it clean.
        
        # Tiny line at top center
        draw.line([(cx - 20, 100), (cx + 20, 100)], fill=colors['accent'], width=4)

    def layout_structure():
        # "The Structure": Asymmetric, vertical axis
        # Vertical line at 1/3
        axis_x = int(WIDTH * 0.33)
        draw.line([(axis_x, 100), (axis_x, HEIGHT - 100)], fill=colors['accent'], width=6)
        
        # Title Right of Axis
        t_size = 150
        t_font = load_font(t_size, font_candidates_title)
        draw.text((axis_x + 80, 300), title, font=t_font, fill=colors['text'])
        
        # Subtitle Left of Axis (Rotated? No, PIL rotation is complex for text placement, keep simple)
        # Just right aligned to axis
        if subtitle:
            s_size = 50
            s_font = load_font(s_size, font_candidates_subtitle)
            sw, sh = get_text_size(subtitle, s_font)
            draw.text((axis_x - sw - 60, 320), subtitle, font=s_font, fill=colors['text'])

    def layout_typo():
        # "Typographic Power": Huge Title, cropped or bleeding
        t_size = 350
        t_font = load_font(t_size, font_candidates_title)
        tw, th = get_text_size(title, t_font)
        
        # Position: Bleeding off right? Or just massive on left
        draw.text((100, HEIGHT // 2 - th // 2), title, font=t_font, fill=colors['text'])
        
        if subtitle:
            s_size = 60
            s_font = load_font(s_size, font_candidates_subtitle)
            draw.text((120, HEIGHT // 2 + th // 2 + 20), subtitle, font=s_font, fill=colors['accent'])

    # --- Background Strategies ---
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def interpolate_color(c1, c2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def draw_gradient(image, c1_hex, c2_hex):
        c1 = hex_to_rgb(c1_hex)
        c2 = hex_to_rgb(c2_hex)
        width, height = image.size
        for y in range(height):
            t = y / height
            color = interpolate_color(c1, c2, t)
            draw.line([(0, y), (width, y)], fill=color)

    def draw_pattern_dots(image, color_hex, spacing=40):
        color = hex_to_rgb(color_hex)
        width, height = image.size
        for x in range(0, width, spacing):
            for y in range(0, height, spacing):
                draw.point((x, y), fill=color)

    def draw_abstract_geometry(image, color_hex):
        # Draw large, subtle geometric shapes
        color = hex_to_rgb(color_hex)
        # Create a transparent overlay
        overlay = Image.new('RGBA', image.size, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        
        # Random large circle
        cx, cy = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        r = random.randint(400, 1000)
        # Low opacity
        fill_color = color + (30,) # 30/255 opacity
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill_color)
        
        # Random diagonal line/rect
        x1 = random.randint(0, WIDTH)
        x2 = random.randint(0, WIDTH)
        d.polygon([(x1, 0), (x2, HEIGHT), (x2+200, HEIGHT), (x1+200, 0)], fill=fill_color)

        image.paste(overlay, (0,0), overlay)

    # --- Execution ---
    
    # 1. Background
    # Randomly choose a background strategy if not specified (implicit for now)
    bg_mode = random.choice(['solid', 'gradient', 'pattern', 'abstract'])
    
    if bg_mode == 'gradient':
        # Subtle gradient based on theme
        if theme == 'light':
            draw_gradient(img, '#FFFFFF', '#F0F0F0')
        elif theme == 'dark':
            draw_gradient(img, '#1A1A1A', '#000000')
        else:
            draw_gradient(img, '#F2F2F2', '#E0E0E0')
    elif bg_mode == 'pattern':
        draw.rectangle([0, 0, WIDTH, HEIGHT], fill=colors['bg'])
        draw_pattern_dots(img, colors['grid'], spacing=30)
    elif bg_mode == 'abstract':
        draw.rectangle([0, 0, WIDTH, HEIGHT], fill=colors['bg'])
        draw_abstract_geometry(img, colors['accent'])
    else:
        # Solid
        draw.rectangle([0, 0, WIDTH, HEIGHT], fill=colors['bg'])

    layouts = {
        'horizon': layout_horizon,
        'void': layout_void,
        'structure': layout_structure,
        'typo': layout_typo
    }
    
    if layout_mode == 'random':
        selected_layout = random.choice(list(layouts.values()))
    else:
        selected_layout = layouts.get(layout_mode, layout_horizon)
        
    # Execute Layout
    selected_layout()
    
    # Post-processing
    add_noise(img, intensity=10) # Apply texture
    
    # Save
    img.save(output_path)
    print(f"Cover image generated: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Hara Style Cover')
    parser.add_argument('--title', required=True, help='Main Title')
    parser.add_argument('--subtitle', help='Subtitle')
    parser.add_argument('--theme', default='light', choices=['light', 'dark', 'grey'], help='Theme')
    parser.add_argument('--output', default='cover.png', help='Output Path')
    parser.add_argument('--layout', default='random', choices=['random', 'horizon', 'void', 'structure', 'typo'], help='Layout Mode')
    
    args = parser.parse_args()
    
    create_hara_cover(args.title, args.subtitle, args.theme, args.output, args.layout)
