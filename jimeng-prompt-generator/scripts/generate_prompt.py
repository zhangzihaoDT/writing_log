import sys

def generate_prompt(keywords):
    # Join keywords (or use the single visual description provided)
    subject = ", ".join(keywords)
    
    # Modifiers optimized for Jimeng 4.1 (targeting Industrial Design Sketch style)
    # The input 'subject' should be a DESCRIPTIVE VISUAL SCENE, not just abstract keywords.
    # Example Input: "A sketch of a Tesla speeding through a tunnel of giant clock gears..."
    
    modifiers = [
        "pixel tech style",                   # 像素科技风格
        "retro computer terminal interface",  # 复古电脑终端界面
        "neon green and purple text",         # 霓虹绿和紫色文字
        "dark grid background",               # 暗色网格背景
        "glitch effect",                      # 故障效果
        "monospace font elements",            # 等宽字体元素
        "command line interface UI",          # 命令行界面UI
        "data stream visualization",          # 数据流可视化
        "hacker aesthetic",                   # 黑客美学
        "cyberpunk atmosphere",               # 赛博朋克氛围
        "high contrast",                      # 高对比度
        "digital art",                        # 数字艺术
        "2.35:1 aspect ratio",                # 2.35:1 宽画幅
        "cinematic widescreen",               # 电影宽屏
        "flat design",                        # 扁平化设计
        "minimalist tech"                     # 极简科技
    ]
    
    # Construct prompt
    # Combine user keywords with the fixed high-quality modifiers
    full_prompt = f"{subject}, {', '.join(modifiers)}"
    
    return full_prompt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_prompt.py <keyword1> <keyword2> ...")
        sys.exit(1)
    
    keywords = sys.argv[1:]
    print(generate_prompt(keywords))
