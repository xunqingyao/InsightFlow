"""报告渲染：Markdown → HTML（只给网页用，命令行仍输出 Markdown）。

报告里的文字来自来源正文与用户问题，因此先做 HTML 转义再交给 Markdown：
这样 `<script>` 只会显示成文本，不会被执行；Markdown 语法不受影响。
"""

from __future__ import annotations

import html

import markdown

_EXTENSIONS = ["tables", "fenced_code", "sane_lists"]


def render_markdown(text: str) -> str:
    escaped = html.escape(text or "", quote=False)
    return markdown.markdown(escaped, extensions=_EXTENSIONS, output_format="html5")