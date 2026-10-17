#!/usr/bin/env python3
"""
Convert Markdown to PDF using markdown and weasyprint
"""
import sys
import os

try:
    import markdown
    from weasyprint import HTML
except ImportError:
    print("Required packages not found. Trying alternative method...")
    sys.exit(1)

# Read markdown file
input_file = 'Presentation_Materials/Gradient_Blending.md'
output_file = 'Presentation_Materials/Gradient_Blending.pdf'

with open(input_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'tables'])

# Add CSS styling
styled_html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Georgia, serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: Courier New, monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            font-style: italic;
            color: #555;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
'''

# Convert to PDF
try:
    HTML(string=styled_html).write_pdf(output_file)
    print(f'✅ PDF created successfully: {output_file}')
except Exception as e:
    print(f'❌ Error creating PDF: {e}')
    sys.exit(1)
