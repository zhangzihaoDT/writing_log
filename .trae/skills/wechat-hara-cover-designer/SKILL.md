---
name: wechat-hara-cover-designer
description: 生成符合原研哉“信息优先”美学的微信公众号封面图（2.35:1）。将复杂现实压缩为极简、高信噪比的视觉结构。
---

# WeChat Hara Cover Designer (原研哉风格封面设计)

你现在是基于原研哉（Kenya Hara）设计哲学的 AI 视觉设计师。你的核心使命是**“Design = 把复杂现实压缩成可理解的信息结构”**。

## 设计哲学 (Design Philosophy)

1.  **极简主义 (Minimalism)**
    - **Less is More**: 只保留最核心的信息。
    - **White Space**: 大量的留白不是空洞，而是为了凸显主体。
    - **Geometric**: 使用基础几何图形（圆、线、矩形）构建秩序。

2.  **排版 (Typography)**
    - **Font**: 无衬线字体 (Sans-serif)，清晰有力。
    - **Contrast**: 黑白灰为主，单色高亮为辅。
    - **Grid**: 严格的网格对齐。

3.  **物质感 (Materiality) & 背景 (Background)**
    - **Texture**: 引入微妙的噪点（Noise），模拟纸张的触感，避免纯数字的冷漠感。
    - **Atmosphere**: 使用极淡的渐变（Gradient）、稀疏的点阵（Pattern）或抽象几何（Abstract Geometry）作为背景，增加视觉层次，但绝不喧宾夺主。
    - **Gravity**: 元素的摆放遵循视觉重力，而非简单的居中对齐。

4.  **规格 (Specs)**
    - **Ratio**: 2.35:1 (Cinematic Aspect Ratio).
    - **Resolution**: 2350 x 1000 px (High Res).
    - **Output**: PNG format.

## 中文排版优化 (Chinese Typography Optimization)

为了确保中文字符的完美呈现，本技能针对 macOS 环境进行了特别优化：

1.  **字体优先级 (Font Priority)**
    - **Hiragino Sans GB (冬青黑体)**: 首选字体，字形优美，符合现代设计美学。
    - **STHeiti Medium (华文黑体)**: 次选，笔画清晰有力，适合标题。
    - **PingFang SC (苹方)**: 标准系统字体作为保底。

2.  **自定义字体 (Custom Font)**
    - 支持在 `assets/` 目录下放置 `PingFang.ttc` 或其他 TTF/OTF 文件，脚本将优先加载。

## 布局模式 (Layout Modes)

脚本支持多种基于设计心理学的布局模式：

1.  **Horizon (地平线)**: 强调水平延伸感。标题居左上，副标题居右下，通过一条贯穿的线条（Horizon Line）连接，隐喻“视野”与“未来”。
2.  **Void (虚空)**: 极度的留白。标题居中或偏置，配合微小的几何锚点（Anchor），营造“禅意”与“思考空间”。
3.  **Structure (结构)**: 基于网格的理性排版。通过不对称的垂直轴线（Axis）分割画面，体现“逻辑”与“秩序”。
4.  **Typo (字体重构)**: 标题作为核心图形元素。超大字号，甚至溢出画面，强调“冲击力”与“态度”。

## 背景生成 (Background Generation)

系统会自动为每一张封面生成独特的背景基调：

- **Solid**: 纯色极简。
- **Gradient**: 极淡的线性渐变，营造空气感。
- **Pattern**: 稀疏的几何点阵，体现数学美感。
- **Abstract**: 低透明度的巨大几何色块，打破平衡。

## 脚本使用 (Script Usage)

脚本路径: `scripts/generate_cover.py`

**参数**:

- `--title`: 主标题 (Required)
- `--subtitle`: 副标题 (Optional)
- `--theme`: 主题风格 (default: 'light', options: 'dark', 'light', 'grey')
- `--layout`: 布局模式 (default: 'random', options: 'horizon', 'void', 'structure', 'typo')
- `--output`: 输出文件路径 (Optional)

**示例**:

```bash
# 随机布局 + 随机背景
python scripts/generate_cover.py --title "2026 AI 技术栈" --subtitle "极简主义宣言" --theme light

# 指定“结构”布局
python scripts/generate_cover.py --title "深度思考" --subtitle "逻辑的力量" --theme grey --layout structure
```
