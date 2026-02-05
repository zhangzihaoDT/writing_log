import os
import sys
import requests
import json
import markdown
import argparse
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_ID = "wx8b7128e0a1cf0d8a"
APP_SECRET = "94ccd6f447e0d2a7948bcb4fd4ac71f5"

# Hara Style Definition (Inline Styles)
STYLES = {
    "body": "font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif; color: #1a1a1a; background-color: #ffffff; line-height: 1.8; font-size: 16px; letter-spacing: 0.05em; margin: 0; padding: 20px; text-align: left;",
    "h1": "font-size: 24px; font-weight: 600; line-height: 1.4; margin-top: 60px; margin-bottom: 30px; letter-spacing: 0.1em; border-bottom: 1px solid #000; padding-bottom: 10px;",
    "h2": "font-size: 20px; font-weight: 600; margin-top: 40px; margin-bottom: 20px; padding-left: 12px; border-left: 4px solid #000;",
    "h3": "font-size: 17px; font-weight: 600; margin-top: 30px; margin-bottom: 15px; color: #333;",
    "p": "margin-bottom: 20px; text-align: left;",
    "strong": "font-weight: 600; color: #000;",
    "em": "font-style: normal; color: #7f7f7f; font-size: 0.95em;",
    "blockquote": "margin: 30px 0; padding: 20px; background-color: #fafafa; border-left: none; color: #555; font-size: 0.95em; position: relative;",
    "pre": "background-color: #f7f7f7; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 20px 0; border: none; line-height: 1.5; white-space: pre; -webkit-overflow-scrolling: touch;",
    "code": "font-family: 'Menlo', 'Monaco', 'Consolas', 'Courier New', monospace; font-size: 13px; color: #333;",
    "a": "color: #1a1a1a; text-decoration: none; border-bottom: 1px solid #ccc;",
    "hr": "border: 0; height: 1px; background: #eee; margin: 60px 0;",
    "ul": "list-style: none; padding-left: 0; margin-bottom: 20px;",
    "ol": "list-style: none; padding-left: 0; margin-bottom: 20px;",
    "li": "margin-bottom: 0; display: flex; align-items: baseline; line-height: 1.5;",
}

def get_access_token(app_id, app_secret):
    """
    Get WeChat Official Account Access Token.
    """
    url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={app_id}&secret={app_secret}"
    response = requests.get(url)
    data = response.json()
    
    if "access_token" in data:
        return data["access_token"]
    else:
        print(f"Error getting access token: {data}")
        sys.exit(1)

def upload_image(access_token, image_path):
    """
    Upload an image to WeChat to get media_id (Permanent Material).
    API: https://api.weixin.qq.com/cgi-bin/material/add_material?access_token=ACCESS_TOKEN&type=image
    """
    url = f"https://api.weixin.qq.com/cgi-bin/material/add_material?access_token={access_token}&type=image"
    
    if not os.path.exists(image_path):
        print(f"Error: Cover image not found at {image_path}")
        sys.exit(1)
        
    print(f"Uploading cover image: {image_path}...")
    try:
        with open(image_path, 'rb') as f:
            files = {'media': f}
            response = requests.post(url, files=files)
            
        data = response.json()
        if 'media_id' in data:
            print(f"Image uploaded successfully. Media ID: {data['media_id']}")
            return data['media_id']
        else:
            print(f"Error uploading image: {data}")
            sys.exit(1)
    except Exception as e:
        print(f"Exception during image upload: {e}")
        sys.exit(1)

def upload_content_image(access_token, image_path):
    """
    Upload an image to be used within the article content.
    API: https://api.weixin.qq.com/cgi-bin/media/uploadimg?access_token=ACCESS_TOKEN
    Returns the URL of the uploaded image.
    """
    url = f"https://api.weixin.qq.com/cgi-bin/media/uploadimg?access_token={access_token}"
    
    if not os.path.exists(image_path):
        print(f"Warning: Content image not found at {image_path}")
        return None
        
    print(f"Uploading content image: {image_path}...")
    try:
        with open(image_path, 'rb') as f:
            files = {'media': f}
            response = requests.post(url, files=files)
            
        data = response.json()
        if 'url' in data:
            print(f"Content image uploaded. URL: {data['url']}")
            return data['url']
        else:
            print(f"Error uploading content image: {data}")
            return None
    except Exception as e:
        print(f"Exception during content image upload: {e}")
        return None

def process_content_images(html_content, access_token):
    """
    Find local images in HTML, upload them to WeChat, and replace src with WeChat URL.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    images = soup.find_all('img')
    
    if not images:
        return html_content
        
    print(f"Found {len(images)} images in content. Processing...")
    
    for img in images:
        src = img.get('src')
        if src and os.path.exists(src):
            # It's a local file
            wechat_url = upload_content_image(access_token, src)
            if wechat_url:
                img['src'] = wechat_url
                # Add responsive style
                existing_style = img.get('style', '')
                img['style'] = f"{existing_style} width: 100% !important; height: auto !important; display: block; margin: 20px 0; border-radius: 4px;".strip()
            else:
                print(f"Failed to upload image: {src}")
    
    return str(soup)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_html_styles(html_content):
    """
    Parse HTML, apply inline styles, and transform lists.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Apply styles to generic tags
    for tag_name, style in STYLES.items():
        for tag in soup.find_all(tag_name):
            existing_style = tag.get('style', '')
            tag['style'] = f"{style} {existing_style}".strip()

    # Special handling: Fix code blocks that were parsed as inline code (common in lists)
    for code in soup.find_all('code'):
        if code.parent.name != 'pre':
            text = code.get_text()
            if '\n' in text:
                # It's a multi-line code span, promote it to a block
                
                # Heuristic: Remove language identifier if it's the first line
                lines = text.split('\n')
                first_line = lines[0].strip().lower()
                common_langs = ['text', 'python', 'py', 'bash', 'sh', 'shell', 'json', 'yaml', 'yml', 'sql', 'javascript', 'js', 'html', 'css']
                if first_line in common_langs:
                    lines = lines[1:]
                    text = '\n'.join(lines)
                    # Update content
                    code.string = text
                
                # Change tag to pre
                code.name = 'pre'
                
                # Merge styles: existing (code) + new (pre)
                # We want pre styles to add box properties, but keep code font properties
                existing_style = code.get('style', '')
                pre_style = STYLES['pre']
                
                # Force block display and merge
                code['style'] = f"{existing_style} {pre_style} display: block; font-family: 'Menlo', 'Monaco', 'Consolas', 'Courier New', monospace; font-size: 13px;".strip()

    # Special handling: Remove margins for paragraphs inside lists (tighten lists)
    for list_tag in soup.find_all(['ul', 'ol']):
        # Handle p tags
        for p in list_tag.find_all('p'):
            existing_style = p.get('style', '')
            # Overwrite margin-bottom and margin-top to 0, and reduce line-height
            p['style'] = f"{existing_style} margin: 0; line-height: 1.5;".strip()
        
        # Handle pre tags (code blocks) inside lists
        for pre in list_tag.find_all('pre'):
            existing_style = pre.get('style', '')
            # Reduce margin for code blocks inside lists
            pre['style'] = f"{existing_style} margin: 8px 0;".strip()

    # Special handling for Blockquotes (add quote mark)
    for bq in soup.find_all('blockquote'):
        # Create quote mark span
        quote_span = soup.new_tag("span")
        quote_span.string = "“"
        quote_span['style'] = "font-size: 40px; color: #e0e0e0; font-family: serif; display: block; line-height: 1; margin-bottom: -10px;"
        bq.insert(0, quote_span)

    # Special handling for Unordered Lists (custom "-")
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li', recursive=False):
            # Create marker
            marker = soup.new_tag("span")
            marker.string = ""
            marker['style'] = "margin-right: 0px; flex-shrink: 0; color: #1a1a1a;"
            
            # Wrap content in a span if not already
            content_span = soup.new_tag("span")
            content_span.extend(li.contents)
            
            li.clear()
            li.append(marker)
            li.append(content_span)

    # Special handling for Ordered Lists (custom "1.", "2.")
    for ol in soup.find_all('ol'):
        for index, li in enumerate(ol.find_all('li', recursive=False)):
            # Create marker
            marker = soup.new_tag("span")
            marker.string = f"{index + 1}."
            marker['style'] = "margin-right: 8px; flex-shrink: 0; color: #1a1a1a;"
            
            # Wrap content
            content_span = soup.new_tag("span")
            content_span.extend(li.contents)
            
            li.clear()
            li.append(marker)
            li.append(content_span)
            
    # Remove <style> tags as we've inlined everything
    for style_tag in soup.find_all('style'):
        style_tag.decompose()
        
    return str(soup)

def convert_markdown_to_html(markdown_content, template_path):
    """
    Convert Markdown to HTML and inject into the Hara template.
    """
    # Use basic markdown conversion
    html_body = markdown.markdown(markdown_content, extensions=['extra', 'sane_lists'])
    
    # Handle task lists (checkboxes) - simplified
    html_body = html_body.replace('[ ]', '[ ]').replace('[x]', '[x]')

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    full_html = template.replace('{{content}}', html_body)
    
    # Process styles
    final_html = process_html_styles(full_html)
    
    return final_html

def push_draft(access_token, title, content, thumb_media_id, author="AI Editor"):
    """
    Push content to WeChat Draft Box.
    API: https://api.weixin.qq.com/cgi-bin/draft/add?access_token=ACCESS_TOKEN
    """
    url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={access_token}"
    
    # We need to extract the <body> content for the API because WeChat wraps it itself?
    # Actually, WeChat expects the 'content' field to be the HTML body.
    # But if we send full <html>, it might strip tags.
    # Best practice is to send what's inside <body>.
    
    soup = BeautifulSoup(content, 'html.parser')
    body_content = "".join([str(x) for x in soup.body.contents]) if soup.body else content
    
    # Construct payload
    article = {
        "title": title,
        "author": author,
        "digest": soup.get_text()[:54] + "...", # Better digest from text
        "content": body_content,
        "content_source_url": "",
        "thumb_media_id": thumb_media_id,
        "need_open_comment": 0,
        "only_fans_can_comment": 0
    }
    
    payload = {
        "articles": [article]
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers=headers)
    data = response.json()
    
    if "media_id" in data:
        print(f"Draft pushed successfully! Draft Media ID: {data['media_id']}")
        return data
    else:
        print(f"Error pushing draft: {data}")
        return data

def main():
    parser = argparse.ArgumentParser(description="Push Markdown to WeChat Draft with Hara Style")
    parser.add_argument("file", help="Path to the Markdown file")
    parser.add_argument("--cover", help="Path to the cover image", required=True)
    
    args = parser.parse_args()
    md_file = args.file
    cover_image = args.cover
    
    # 1. Parse content
    print(f"Reading {md_file}...")
    if not os.path.exists(md_file):
        print(f"Error: Markdown file not found at {md_file}")
        sys.exit(1)

    content_md = read_file(md_file)
    
    # Extract title from filename
    title = os.path.basename(md_file).replace('.md', '').replace('_', ' ')
    
    # 2. Convert to Hara Style HTML
    template_path = os.path.join(os.path.dirname(__file__), '../assets/template.html')
    print("Converting to Hara Style HTML...")
    if not os.path.exists(template_path):
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)
        
    html_content = convert_markdown_to_html(content_md, template_path)
    
    # Save the HTML for inspection
    output_html = md_file.replace('.md', '_wechat.html')
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated HTML saved to: {output_html}")
    
    # 3. Push to WeChat
    print("Getting Access Token...")
    token = get_access_token(APP_ID, APP_SECRET)
    print(f"Access Token retrieved.")
    
    # 4. Upload Cover Image
    thumb_media_id = upload_image(token, cover_image)
    
    # 4.5 Process Content Images
    print("Processing content images...")
    html_content = process_content_images(html_content, token)
    
    # 5. Push Draft
    print(f"Pushing draft '{title}'...")
    push_draft(token, title, html_content, thumb_media_id)

if __name__ == "__main__":
    main()
