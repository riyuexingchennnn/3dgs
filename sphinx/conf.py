# -*- coding: utf-8 -*-
"""Sphinx 配置：把 Markdown 教程构建为带左侧目录导航的静态站点。"""

project = "三维高斯溅射（3DGS）完全教程"
author = "日月星辰"
copyright = "2026, 日月星辰"

language = "zh_CN"
locale_dirs = []
gettext_compact = False

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

# ---- Markdown 解析 ----
source_suffix = {".md": "markdown"}
myst_enable_extensions = [
    "dollarmath",   # $...$ 与 $$...$$
    "amsmath",      # \begin{align} 等环境
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3

# ---- 公式渲染（MathJax）----
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
        "processEscapes": True,
    },
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre", "code"],
    },
}

# ---- 主题：RTD，左侧目录导航，强制白色 ----
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "3DGS 完全教程"
html_short_title = "3DGS 教程"

# ---- 输出精简 ----
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
html_last_updated_fmt = ""
html_use_smartypants = False

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header", "image.not_readable"]
