---
name: jimeng-prompt-generator
description: Generate optimized image generation prompts for Jimeng 4.1 (即梦) to create WeChat Official Account cover images. Use this skill when the user wants to generate a cover image prompt based on keywords.
---

# Jimeng Prompt Generator

This skill generates prompts optimized for Jimeng 4.1 AI art generator, specifically tailored for **Industrial Design Sketch** style WeChat Official Account cover images (2.35:1 aspect ratio).

It emphasizes **Semantic Understanding and Visual Translation**: converting abstract concepts into concrete, sketchable visual elements before generating the final prompt.

## Style Description
- **Type**: Industrial Design Sketch / Automotive Concept Art
- **Visuals**: Pencil and marker drawing, vintage parchment paper background, technical annotations (arrows, details), rough sketch lines.
- **Vibe**: Professional, technical, vintage, artistic.

## Usage

1.  **Analyze & Translate**:
    -   Do NOT just pass abstract keywords (e.g., "Growth", "Future").
    -   **Translate** abstract concepts into concrete visual scenes (e.g., "A futuristic seedling breaking through metal gears", "A glowing city on the horizon").
    -   Combine these visuals into a cohesive scene description.
2.  **Execute Script**:
    -   Run the python script `scripts/generate_prompt.py` with the **visual description** as arguments.
    ```bash
    python3 jimeng-prompt-generator/scripts/generate_prompt.py "Visual Description of the scene"
    ```
3.  **Output**:
    -   The script will append the fixed Industrial Design style modifiers to your visual description.
    -   Return the final prompt to the user.

## Example

**User Input:** "Create a prompt for: Time's Game, Tesla, Credit Expansion"

**1. Analysis (Internal Thought):**
-   *Time's Game* -> Clock gears, hourglass, motion blur.
-   *Tesla* -> Cybertruck silhouette, electric arcs, battery cells.
-   *Credit Expansion* -> Expanding bubbles, rising graphs, overflowing coins.
-   *Combined Visual*: "A sketch of a Tesla speeding through a tunnel of giant clock gears, creating a wake of golden coins that are expanding and dissolving into smoke."

**2. Command:**
```bash
python3 jimeng-prompt-generator/scripts/generate_prompt.py "A sketch of a Tesla speeding through a tunnel of giant clock gears, creating a wake of golden coins that are expanding and dissolving into smoke"
```

**3. Output:**
"A sketch of a Tesla speeding through a tunnel of giant clock gears, creating a wake of golden coins that are expanding and dissolving into smoke, industrial design sketch, automotive concept drawing..."

## Example

**User Input:** "Create a prompt for: Tesla, Future, Speed"

**Command:**

```bash
python3 jimeng-prompt-generator/scripts/generate_prompt.py "Tesla" "Future" "Speed"
```

**Output:**
"Tesla, Future, Speed, futuristic sci-fi style, motion blur, cinematic lighting, 8k resolution, wide angle, masterpiece"
