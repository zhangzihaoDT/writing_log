import sys

def generate_prompt(keywords):
    # Join keywords (or use the single visual description provided)
    subject = ", ".join(keywords)
    
    # Modifiers optimized for Jimeng 4.1 (targeting Industrial Design Sketch style)
    # The input 'subject' should be a DESCRIPTIVE VISUAL SCENE, not just abstract keywords.
    # Example Input: "A sketch of a Tesla speeding through a tunnel of giant clock gears..."
    
    modifiers = [
        "industrial design sketch",           # 工业设计手绘
        "automotive concept drawing",         # 汽车概念图
        "pencil and marker style",            # 铅笔马克笔风格
        "vintage parchment paper background", # 复古羊皮纸背景
        "technical annotations and arrows",   # 技术标注和箭头
        "rough sketch lines",                 # 粗糙草图线条
        "detailed mechanical structure",      # 详细机械结构
        "concept art masterpiece",            # 概念艺术杰作
        "high contrast",                      # 高对比度
        "warm vintage tones",                 # 温暖复古色调
        "highly detailed",                    # 高细节
        "2.35:1 aspect ratio",                # 2.35:1 宽画幅
        "cinematic widescreen",               # 电影宽屏
        "panoramic view"                      # 全景视图
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
