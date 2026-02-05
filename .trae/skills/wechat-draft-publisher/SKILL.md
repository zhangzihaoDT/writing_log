---
name: wechat-draft-publisher
description: Publish Markdown content to WeChat Official Account (公众号) drafts. Use this skill when the user wants to publish an article to WeChat.
---

# WeChat Draft Publisher

This skill allows publishing content to WeChat Official Account drafts using the official WeChat API.

## Prerequisites

1.  **Python Environment**: Ensure `requests` is installed (`pip install requests`).
2.  **WeChat Credentials**: `APP_ID` and `APP_SECRET` are required.
3.  **Cover Image**: A local image file path is required for the cover.

## Usage Flow

1.  **Prepare Content**:
    *   Read the Markdown file provided by the user.
    *   Convert the Markdown content to HTML. You (Claude) should perform this conversion, ensuring the HTML is clean and suitable for WeChat.
        *   Use `<h2>` and `<h3>` for headers.
        *   Use `<p>` for paragraphs.
        *   **Important**: WeChat articles do not support external CSS classes well. Use inline styles if specific formatting is requested.
    *   Save the HTML content to a temporary file (e.g., `temp_article.html`).

2.  **Prepare Metadata**:
    *   Extract or ask for the `Title` (default to filename), `Author`, and `Digest` (summary).
    *   Ensure a `Cover Image` path is available. **This is mandatory.** If the user hasn't provided one, ask for it.

3.  **Execute Script**:
    Run the `scripts/publish.py` script with the required arguments.

    ```bash
    python3 wechat-draft-publisher/scripts/publish.py \
      --app_id <APP_ID> \
      --app_secret <APP_SECRET> \
      --html_file <PATH_TO_HTML> \
      --title "<TITLE>" \
      --cover_image <PATH_TO_IMAGE> \
      --author "<AUTHOR>" \
      --digest "<DIGEST>"
    ```

4.  **Cleanup**:
    *   Delete the temporary HTML file after success.

## Error Handling

*   If `APP_SECRET` is missing, ask the user to provide it.
*   If `requests` module is missing, suggest running `pip install requests`.
