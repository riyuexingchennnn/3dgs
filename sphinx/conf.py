# -*- coding: utf-8 -*-
"""Sphinx 配置：构建带左侧目录导航的 3DGS 教程站点，外观对齐 ROS 2 文档。"""

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
    "dollarmath",
    "amsmath",
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

# ---- 主题：RTD（ROS 2 文档同款），深色侧边栏 + 白色正文 ----
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_title = "3DGS 教程文档"
html_short_title = "3DGS 教程"

# ---- 右上角 Edit on GitHub ----
html_context = {
    "display_github": True,
    "github_user": "riyuexingchennnn",
    "github_repo": "3dgs",
    "github_version": "master",
    "conf_py_path": "/sphinx/",
}

# ---- 输出 ----
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
html_last_updated_fmt = ""
html_use_smartypants = False

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header", "image.not_readable"]
