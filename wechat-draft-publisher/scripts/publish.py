import argparse
import requests
import json
import os
import sys

def get_access_token(app_id, app_secret):
    url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={app_id}&secret={app_secret}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if 'access_token' not in data:
            print(f"Error getting access token: {data}")
            sys.exit(1)
        return data['access_token']
    except Exception as e:
        print(f"Network error getting token: {e}")
        sys.exit(1)

def upload_image(access_token, image_path):
    url = f"https://api.weixin.qq.com/cgi-bin/material/add_material?access_token={access_token}&type=image"
    try:
        with open(image_path, 'rb') as f:
            files = {'media': f}
            resp = requests.post(url, files=files)
            data = resp.json()
            if 'media_id' not in data:
                print(f"Error uploading image: {data}")
                sys.exit(1)
            return data['media_id']
    except FileNotFoundError:
        print(f"Cover image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error uploading image: {e}")
        sys.exit(1)

def create_draft(access_token, articles):
    url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={access_token}"
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    # Ensure proper JSON encoding for Chinese characters
    data = json.dumps({'articles': articles}, ensure_ascii=False).encode('utf-8')
    try:
        resp = requests.post(url, data=data, headers=headers)
        result = resp.json()
        if 'media_id' in result:
            print(f"Draft created successfully! Media ID: {result['media_id']}")
        else:
            print(f"Error creating draft: {result}")
    except Exception as e:
        print(f"Error posting draft: {e}")

def main():
    parser = argparse.ArgumentParser(description="Publish HTML to WeChat Draft")
    parser.add_argument('--app_id', required=True, help="WeChat AppID")
    parser.add_argument('--app_secret', required=True, help="WeChat AppSecret")
    parser.add_argument('--html_file', required=True, help="Path to the HTML content file")
    parser.add_argument('--title', required=True, help="Article Title")
    parser.add_argument('--cover_image', required=True, help="Path to cover image (required)")
    parser.add_argument('--author', default="", help="Author Name")
    parser.add_argument('--digest', default="", help="Article Digest/Summary")

    args = parser.parse_args()

    if not os.path.exists(args.html_file):
        print(f"HTML file not found: {args.html_file}")
        sys.exit(1)

    # Read HTML content
    with open(args.html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Getting Access Token for AppID: {args.app_id}...")
    token = get_access_token(args.app_id, args.app_secret)
    
    print(f"Uploading cover image: {args.cover_image}...")
    media_id = upload_image(token, args.cover_image)

    article = {
        "title": args.title,
        "author": args.author,
        "digest": args.digest,
        "content": content,
        "thumb_media_id": media_id
    }

    print("Creating draft...")
    create_draft(token, [article])

if __name__ == "__main__":
    main()
