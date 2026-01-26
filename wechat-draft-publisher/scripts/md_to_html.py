import markdown
import sys
import os

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Use extensions for better formatting
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Add some basic styling for WeChat
    # WeChat requires inline styles for best results, but we can wrap it in a div
    # and hope for the best, or use a simple post-processor.
    # For now, let's just wrap it in a section with some basic styling.
    
    styled_html = f"""
    <div style="font-size: 16px; line-height: 1.6; color: #333; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;">
        {html}
    </div>
    """
    
    # Simple post-processing to add style to headers
    styled_html = styled_html.replace('<h1>', '<h1 style="font-size: 22px; font-weight: bold; margin-top: 20px; margin-bottom: 10px;">')
    styled_html = styled_html.replace('<h2>', '<h2 style="font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; border-left: 4px solid #E82127; padding-left: 10px;">')
    styled_html = styled_html.replace('<h3>', '<h3 style="font-size: 16px; font-weight: bold; margin-top: 15px; margin-bottom: 8px;">')
    styled_html = styled_html.replace('<p>', '<p style="margin-bottom: 15px; text-align: justify;">')
    styled_html = styled_html.replace('<blockquote>', '<blockquote style="border-left: 4px solid #ddd; padding-left: 10px; color: #666; margin: 15px 0;">')
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print(f"Converted HTML saved to: {html_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 md_to_html.py <input_md> <output_html>")
        sys.exit(1)
    
    convert_md_to_html(sys.argv[1], sys.argv[2])
