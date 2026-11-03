#!/usr/bin/env python3
"""
Markdown to HTML Converter for DeerFlow Reports
Convert markdown reports to beautifully styled HTML pages.
"""

import sys
import re
from pathlib import Path

# Simple markdown to HTML conversion
def convert_markdown_to_html(markdown_content: str, title: str = "Research Report") -> str:
    """Convert markdown content to HTML with styling."""

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.8;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px 50px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            font-size: 2.2em;
            color: #1a1a1a;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #2563eb;
        }}
        h2 {{
            font-size: 1.6em;
            color: #1a1a1a;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}
        h3 {{
            font-size: 1.3em;
            color: #2563eb;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        p {{
            margin-bottom: 16px;
            text-align: justify;
        }}
        ul, ol {{
            margin-left: 24px;
            margin-bottom: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
        }}
        pre code {{
            background: none;
            padding: 0;
            color: inherit;
        }}
        blockquote {{
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            padding: 15px 20px;
            margin: 20px 0;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #e5e7eb;
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 30px 0;
        }}
        .meta {{
            color: #64748b;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .confidence {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .confidence-high {{ background: #dcfce7; color: #166534; }}
        .confidence-medium {{ background: #fef3c7; color: #92400e; }}
        .confidence-low {{ background: #fee2e2; color: #991b1b; }}
        .citation {{
            color: #2563eb;
            font-size: 0.85em;
        }}
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 20px; }}
            h1 {{ font-size: 1.6em; }}
            h2 {{ font-size: 1.3em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
{process_content(markdown_content)}
    </div>
</body>
</html>'''

    return html

def process_content(content: str) -> str:
    """Process markdown content to HTML."""
    lines = content.split('\n')
    result = []
    in_code_block = False
    in_list = False

    for line in lines:
        # Code blocks
        if line.startswith('```'):
            if in_code_block:
                result.append('</code></pre>')
            else:
                lang = line[3:].strip()
                result.append(f'<pre><code class="language-{lang}">')
            in_code_block = not in_code_block
            continue

        if in_code_block:
            result.append(line)
            continue

        # Headers
        if line.startswith('# '):
            result.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            result.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            result.append(f'<h3>{line[4:]}</h3>')
        # Horizontal rule
        elif line.strip() == '---':
            result.append('<hr>')
        # Blockquote
        elif line.startswith('> '):
            result.append(f'<blockquote>{line[2:]}</blockquote>')
        # Unordered list
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{line.strip()[2:]}</li>')
        # Ordered list
        elif re.match(r'^\d+\.\s', line):
            if not in_list:
                result.append('<ol>')
                in_list = True
            cleaned = re.sub(r"^\d+\.\s", "", line)
            result.append(f'<li>{cleaned}</li>')
        # End list
        elif in_list and line.strip() and not line.strip().startswith(('-', '*', '1.', '2.', '3.')):
            if in_list:
                result.append('</ul>' if '<ul>' in ''.join(result[-10:]) else '</ol>')
                in_list = False
            result.append(f'<p>{format_inline(line)}</p>')
        # Empty line
        elif not line.strip():
            if in_list:
                result.append('</ul>' if '<ul>' in ''.join(result[-10:]) else '</ol>')
                in_list = False
            continue
        # Regular paragraph
        else:
            result.append(f'<p>{format_inline(line)}</p>')

    if in_list:
        result.append('</ul>' if '<ul>' in ''.join(result[-10:]) else '</ol>')

    return '\n'.join(result)

def format_inline(text: str) -> str:
    """Format inline elements like bold, italic, links."""
    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    # Bold **text** or __text__
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', text)
    # Italic *text* or _text_
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    text = re.sub(r'_([^_]+)_', r'<em>\1</em>', text)
    # Inline code `code`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Citation [citation:Title](url)
    text = re.sub(r'\[citation:([^\]]+)\]\(([^)]+)\)', r'<span class="citation">[\1](\2)</span>', text)
    return text

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python md2html.py <input.md> [output.html]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.md', '.html')

    # Read markdown
    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Extract title from first H1 or filename
    title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
    title = title_match.group(1) if title_match else Path(input_file).stem

    # Convert
    html_content = convert_markdown_to_html(md_content, title)

    # Write HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {output_file}")
