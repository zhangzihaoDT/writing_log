import markdown
import sys
import os
import re

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Use extensions for better formatting
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Configuration
    highlight_color = "#ff7faa"
    font_size_title = "20px"
    font_size_content = "16px"
    line_height = "1.6"
    paragraph_spacing = "16px"
    
    # Base container
    styled_html = f"""
    <div style="font-size: {font_size_content}; line-height: {line_height}; color: #333; text-align: left; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;">
        {html}
    </div>
    """
    
    # Post-processing styles
    
    # Headers
    # H1 & H2: 20px, bold, dark color, with highlighted left border
    header_style = f'style="font-size: {font_size_title}; font-weight: bold; color: #333; margin-top: 24px; margin-bottom: 16px; border-left: 4px solid {highlight_color}; padding-left: 10px;"'
    styled_html = styled_html.replace('<h1>', f'<h1 {header_style}>')
    styled_html = styled_html.replace('<h2>', f'<h2 {header_style}>')
    
    # H3: 18px (slightly distinct), dark color
    h3_style = f'style="font-size: 18px; font-weight: bold; color: #333; margin-top: 20px; margin-bottom: 12px;"'
    styled_html = styled_html.replace('<h3>', f'<h3 {h3_style}>')
    
    # Paragraphs: 16px, spacing 16px, left align
    p_style = f'style="font-size: {font_size_content}; margin-bottom: {paragraph_spacing}; text-align: left;"'
    styled_html = styled_html.replace('<p>', f'<p {p_style}>')
    
    # Blockquotes
    blockquote_style = 'style="border-left: 4px solid #ddd; padding-left: 10px; color: #666; margin: 16px 0;"'
    styled_html = styled_html.replace('<blockquote>', f'<blockquote {blockquote_style}>')
    
    # Strong/Bold: Apply highlight color
    strong_style = f'style="color: {highlight_color}; font-weight: bold;"'
    styled_html = styled_html.replace('<strong>', f'<strong {strong_style}>')
    
    # HR: Replace with extra spacing (empty div or margin)
    # Instead of a visible line, we use a spacer
    hr_style = 'style="border: none; margin-top: 40px; margin-bottom: 40px;"'
    styled_html = styled_html.replace('<hr>', f'<hr {hr_style}>')
    # Or replace it with a spacer div entirely if hr still shows a line in some renderers (though border:none should work)
    # styled_html = styled_html.replace('<hr>', f'<div style="height: 40px;"></div>') 
    # Let's stick to styled HR for semantic correctness but visual spacing
    
    # Lists: Custom bullet point "-" and Circle Numbers "①"
    # Process lists with a stack to handle nesting and distinguish UL/OL
    def process_lists(html_content):
        circle_nums = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
        
        # Styles
        ul_style = 'style="list-style-type: none; padding-left: 0; margin-bottom: 16px;"'
        ol_style = 'style="list-style-type: none; padding-left: 0; margin-bottom: 16px;"'
        
        # LI styles
        # UL LI: Short dash
        li_ul_style = 'style="margin-bottom: 8px; padding-left: 20px; text-indent: -14px;"'
        # OL LI: Circle number (Need more indent)
        li_ol_style = 'style="margin-bottom: 8px; padding-left: 34px; text-indent: -34px;"'
        
        pattern = re.compile(r'(</?(?:ul|ol|li)>)')
        parts = pattern.split(html_content)
        
        output = []
        stack = [] # 'ul' or 'ol'
        ol_counters = [] # list of ints
        
        for part in parts:
            if part == '<ul>':
                stack.append('ul')
                output.append(f'<ul {ul_style}>')
            elif part == '</ul>':
                if stack: stack.pop()
                output.append('</ul>')
            elif part == '<ol>':
                stack.append('ol')
                ol_counters.append(0)
                output.append(f'<ol {ol_style}>')
            elif part == '</ol>':
                if stack: stack.pop()
                if ol_counters: ol_counters.pop()
                output.append('</ol>')
            elif part == '<li>':
                parent = stack[-1] if stack else 'ul'
                
                # Handle Loose Lists: If LI contains <p> with margin-bottom: 16px, remove that margin.
                # The content of LI is in the NEXT part(s)? No, 'parts' is split by tags.
                # Wait, re.split(r'(</?(?:ul|ol|li)>)') splits by tags.
                # So the structure is: [..., '<li>', 'Content...', '</li>', ...]
                # So we are currently at '<li>'. The content is in the NEXT iteration?
                # No! My loop iterates over parts.
                # If part == '<li>', I append the opening tag.
                # The content is processed in the 'else' block (part is content).
                # But I need to modify the content if it's inside a LI.
                # This approach of splitting and iterating is tricky if I need to modify content based on parent.
                
                if parent == 'ul':
                    output.append(f'<li {li_ul_style}>- ')
                else:
                    # OL
                    count = ol_counters[-1] if ol_counters else 0
                    if ol_counters: ol_counters[-1] += 1
                    
                    num_str = circle_nums[count] if count < len(circle_nums) else f"{count+1}."
                    # Styled number
                    num_span = f'<span style="font-weight: bold; color: #333; margin-right: 6px;">{num_str}</span>'
                    output.append(f'<li {li_ol_style}>{num_span}')
            elif part == '</li>':
                output.append('</li>')
            else:
                # Content or other tags
                # If we are inside a LI, we should remove margin from <p> tags
                if stack and (stack[-1] == 'ul' or stack[-1] == 'ol'):
                     # Check if we are directly inside LI? 
                     # The stack tracks UL/OL. It doesn't track LI.
                     # But strictly speaking, text inside UL/OL MUST be inside LI.
                     # However, 'parts' includes '<li>', then content, then '</li>'.
                     # When we are processing content, we are between <li> and </li>.
                     # So yes, we can modify 'part'.
                     
                     # Replace margin-bottom: 16px with margin-bottom: 0 for p tags inside lists
                     part = part.replace('margin-bottom: 16px;', 'margin-bottom: 0;')
                
                output.append(part)
                
        return "".join(output)

    styled_html = process_lists(styled_html)
    
    # Fix nested lists if any (optional cleanup)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print(f"Converted HTML saved to: {html_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 md_to_html.py <input_md> <output_html>")
        sys.exit(1)
    
    convert_md_to_html(sys.argv[1], sys.argv[2])
