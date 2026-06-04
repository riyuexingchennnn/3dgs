"""
Chapter 6 Figures for 3DGS book.
Generates 6 figures saved to docs/pic/ch6_fig*.png
All labels in Chinese. Font: Noto Sans CJK SC.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Arc, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter, uniform_filter
import warnings
warnings.filterwarnings('ignore')

# ── Font setup ──────────────────────────────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp = FontProperties(fname=FONT_PATH)
BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fp_bold = FontProperties(fname=BOLD_PATH)

# Register the font files explicitly so matplotlib finds the family
from matplotlib.font_manager import fontManager as _fm
for _p in [FONT_PATH, BOLD_PATH]:
    _fm.addfont(_p)
# Use the family name as registered by the .ttc (JP variant = same glyphs)
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/home/root123/Documents/3dgs/docs/pic'
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def ctext(ax, x, y, s, bold=False, **kw):
    """Draw Chinese text with Noto Sans CJK SC."""
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    kw['fontproperties'] = fp_bold if bold else fp
    return ax.text(x, y, s, **kw)


def rbox(ax, x, y, w, h, label, fc='#AED6F1', ec='#1A5276',
         fontsize=10, color='#1A5276', radius=0.05, bold=False, zorder=3):
    """Rounded rectangle with centered text."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f'round,pad={radius}',
                       fc=fc, ec=ec, lw=1.8, zorder=zorder)
    ax.add_patch(p)
    ctext(ax, x, y, label, bold=bold, fontsize=fontsize,
          color=color, zorder=zorder + 1)
    return p


def arr(ax, x0, y0, x1, y1, color='#444444', lw=1.8, shrink=3, style='->'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, shrinkA=shrink, shrinkB=shrink))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — 3DGS整体训练流程图
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FBFF')
    ax.set_facecolor('#F8FBFF')

    ctext(ax, 8.5, 8.6, '3DGS 整体训练流程图',
          bold=True, fontsize=17, color='#1A5276')

    # ── Top pipeline ────────────────────────────────────────────────────────
    pipeline = [
        (1.3,  6.2, '多视角\n图像',   '#D5E8D4', '#27AE60'),
        (3.4,  6.2, 'COLMAP\n特征匹配 & SfM', '#FFF2CC', '#B8860B'),
        (5.8,  6.2, '稀疏\n点云',    '#FFF2CC', '#B8860B'),
        (7.9,  6.2, '初始化\n高斯基元', '#DAE8FC', '#2874A6'),
    ]
    for x, y, lbl, fc, ec in pipeline:
        rbox(ax, x, y, 1.9, 1.0, lbl, fc=fc, ec=ec,
             fontsize=10, color=ec, bold=True)

    for x0, x1 in [(2.25, 2.45), (4.35, 4.85), (6.75, 6.95)]:
        arr(ax, x0, 6.2, x1, 6.2, color='#555', lw=2)
    arr(ax, 8.85, 6.2, 9.6, 6.2, color='#555', lw=2)

    # ── Training loop outer box ─────────────────────────────────────────────
    loop_bg = FancyBboxPatch((9.5, 1.5), 7.0, 5.5,
                              boxstyle='round,pad=0.12',
                              fc='#FEF9F0', ec='#C0392B', lw=2.5, zorder=1)
    ax.add_patch(loop_bg)
    ctext(ax, 13.0, 6.75, '迭代训练循环（30,000 步）',
          bold=True, fontsize=12, color='#C0392B')

    # Steps in loop — two rows
    row1 = [
        (10.7, 5.5, '前向渲染', '#EBDEF0', '#7D3C98'),
        (13.0, 5.5, '计算 Loss\n(L1 + SSIM)', '#EBDEF0', '#7D3C98'),
        (15.3, 5.5, '反向传播\n梯度计算', '#EBDEF0', '#7D3C98'),
    ]
    row2 = [
        (10.7, 3.2, 'ADC\n自适应密度控制', '#FDEBD0', '#CA6F1E'),
        (13.0, 3.2, '参数更新\n(Adam 优化器)', '#D5F5E3', '#1E8449'),
        (15.3, 3.2, '输出场景\n(.ply 文件)', '#D5E8D4', '#27AE60'),
    ]
    for x, y, lbl, fc, ec in row1 + row2:
        rbox(ax, x, y, 1.9, 1.0, lbl, fc=fc, ec=ec,
             fontsize=9.5, color=ec, bold=False, zorder=3)

    # Arrows inside loop
    arr(ax, 11.65, 5.5, 12.05, 5.5)            # render -> loss
    arr(ax, 13.95, 5.5, 14.35, 5.5)            # loss -> backprop
    arr(ax, 15.3, 5.0, 15.3, 3.7)             # backprop -> output (via param)
    arr(ax, 14.35, 3.2, 13.95, 3.2)           # param update <- backprop
    arr(ax, 13.0, 3.7, 13.0, 5.0)             # param update -> render (up)
    arr(ax, 12.05, 3.2, 11.65, 3.2, color='#CA6F1E')  # param -> ADC
    arr(ax, 10.7, 3.7, 10.7, 5.0, color='#CA6F1E')    # ADC -> render

    # Output scene external
    rbox(ax, 15.3, 1.3, 1.9, 0.75, '最终渲染\n结果', fc='#D5E8D4',
         ec='#27AE60', fontsize=9.5, color='#27AE60', bold=True, zorder=3)
    arr(ax, 15.3, 2.7, 15.3, 1.68, color='#27AE60', lw=2)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend = [
        ('#D5E8D4', '#27AE60', '输入/输出'),
        ('#FFF2CC', '#B8860B', '预处理'),
        ('#DAE8FC', '#2874A6', '初始化'),
        ('#FEF9F0', '#C0392B', '训练循环'),
        ('#EBDEF0', '#7D3C98', '训练步骤'),
        ('#FDEBD0', '#CA6F1E', 'ADC控制'),
    ]
    for i, (fc, ec, lbl) in enumerate(legend):
        bx = 0.5 + i * 2.6
        p = FancyBboxPatch((bx, 0.4), 0.5, 0.38,
                           boxstyle='round,pad=0.05',
                           fc=fc, ec=ec, lw=1.5)
        ax.add_patch(p)
        ctext(ax, bx + 0.85, 0.59, lbl, fontsize=9,
              ha='left', color='#333')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ch6_fig1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — 高斯基元参数示意
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('#FDFEFE')

    # ── Left: 2D depiction of 3D ellipsoid ──────────────────────────────────
    ax_l = fig.add_axes([0.02, 0.08, 0.52, 0.88])
    ax_l.set_xlim(-4.5, 4.5)
    ax_l.set_ylim(-4.0, 4.5)
    ax_l.set_aspect('equal')
    ax_l.axis('off')
    ax_l.set_facecolor('#EAF2FF')
    ctext(ax_l, 0, 4.1, '高斯基元几何参数示意', bold=True,
          fontsize=14, color='#1A5276')

    # Body of ellipsoid
    ell = Ellipse((0, 0), width=4.0, height=2.2, angle=20,
                  fc='#85C1E9', ec='#2874A6', lw=2.5, alpha=0.5, zorder=2)
    ax_l.add_patch(ell)
    # Equator (simulate 3D depth)
    arc_f = Arc((0, 0), 4.0, 0.75, angle=20, theta1=0, theta2=180,
                color='#2874A6', lw=1.8, ls='--', zorder=3)
    arc_b = Arc((0, 0), 4.0, 0.75, angle=20, theta1=180, theta2=360,
                color='#2874A6', lw=1.8, zorder=3)
    ax_l.add_patch(arc_f)
    ax_l.add_patch(arc_b)

    # Center point
    ax_l.plot(0, 0, 'o', color='#E74C3C', ms=9, zorder=6)

    # Principal axes
    axes_def = [
        ((1.8, 0.8),  '#E74C3C', 's₁ 主轴1（最长）'),
        ((-0.5, 1.1), '#27AE60', 's₂ 主轴2（中等）'),
        ((0.6, -1.2), '#8E44AD', 's₃ 主轴3（最短）'),
    ]
    for (tx, ty), col, lbl in axes_def:
        ax_l.annotate('', xy=(tx, ty), xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color=col, lw=3.0))
        ax_l.text(tx * 1.25, ty * 1.25, lbl, fontproperties=fp,
                  fontsize=10, color=col, ha='center', va='center')

    # Rotation arc annotation
    rot_arc = Arc((0, 0), 1.4, 1.4, angle=0, theta1=25, theta2=155,
                  color='#F39C12', lw=2.5, zorder=4)
    ax_l.add_patch(rot_arc)
    ax_l.annotate('', xy=(-0.65, 0.58), xytext=(-0.50, 0.70),
                  arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2))
    ax_l.text(-1.2, 1.0, 'q 旋转方向', fontproperties=fp,
              fontsize=10, color='#F39C12', ha='center')

    # Position label
    ax_l.text(0.2, -0.35, 'μ (质心位置)', fontproperties=fp,
              fontsize=11, color='#E74C3C', ha='left', fontweight='bold')
    # Opacity label
    ax_l.text(0, -2.5, 'α = 不透明度（sigmoid 激活，控制可见性）',
              fontproperties=fp, fontsize=10, color='#1A5276', ha='center',
              bbox=dict(fc='#D6EAF8', ec='#2874A6', boxstyle='round,pad=0.3'))

    # ── Right: parameter table ───────────────────────────────────────────────
    ax_r = fig.add_axes([0.56, 0.08, 0.42, 0.88])
    ax_r.set_xlim(0, 10)
    ax_r.set_ylim(0, 10)
    ax_r.axis('off')
    ax_r.set_facecolor('#FDFEFE')
    ctext(ax_r, 5, 9.5, '可学习参数列表', bold=True, fontsize=14, color='#1A5276')

    params = [
        ('μ', '位置', '3 个标量  (x, y, z)',
         '#FADBD8', '#C0392B'),
        ('q', '旋转四元数', '4 个标量  (w, x, y, z)，单位化',
         '#FDEBD0', '#CA6F1E'),
        ('s', '缩放系数', '3 个标量  (s₁, s₂, s₃)，经 exp 激活',
         '#FEF9E7', '#B7950B'),
        ('α', '不透明度', '1 个标量，经 sigmoid 激活',
         '#D5F5E3', '#1E8449'),
        ('f_SH', '球谐颜色系数', '48 个标量（0~3 阶，RGB 各 16）',
         '#EBF5FB', '#1A5276'),
    ]
    y0 = 8.7
    for sym, name, desc, fc, ec in params:
        p = FancyBboxPatch((0.3, y0 - 0.85), 9.4, 1.30,
                           boxstyle='round,pad=0.08',
                           fc=fc, ec=ec, lw=1.8, zorder=2)
        ax_r.add_patch(p)
        ax_r.text(0.9, y0 - 0.05, sym, fontproperties=fp_bold,
                  fontsize=14, color=ec, ha='left', va='center')
        ax_r.text(1.9, y0 - 0.05, name, fontproperties=fp_bold,
                  fontsize=11, color=ec, ha='left', va='center')
        ax_r.text(1.9, y0 - 0.60, desc, fontproperties=fp,
                  fontsize=9.5, color='#444', ha='left', va='center')
        y0 -= 1.65

    # Total
    p_tot = FancyBboxPatch((0.3, 0.35), 9.4, 0.85,
                           boxstyle='round,pad=0.08',
                           fc='#D6EAF8', ec='#1A5276', lw=2.5)
    ax_r.add_patch(p_tot)
    ctext(ax_r, 5, 0.78, '合计：59 个可学习参数 / 每个高斯基元',
          bold=True, fontsize=12, color='#1A5276')

    out = os.path.join(OUT_DIR, 'ch6_fig2.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FDFEFE')
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — 3D高斯投影到2D
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#FDFEFE')
    ctext(ax, 7, 7.55, '3D 高斯投影至 2D 图像平面',
          bold=True, fontsize=16, color='#1A5276')

    # ── Left panel: 3D world ────────────────────────────────────────────────
    world = FancyBboxPatch((0.2, 0.5), 5.8, 6.5,
                           boxstyle='round,pad=0.12',
                           fc='#EBF5FB', ec='#2874A6', lw=2)
    ax.add_patch(world)
    ctext(ax, 3.1, 6.75, '三维世界空间', bold=True, fontsize=12, color='#1A5276')

    # Camera frustum
    frustum = plt.Polygon(
        [[1.0, 1.2], [0.5, 5.8], [5.5, 5.8], [5.0, 1.2]],
        fc='#D6EAF8', ec='#1A5276', lw=1.5, alpha=0.35)
    ax.add_patch(frustum)
    ax.plot(3.0, 0.9, 's', color='#1A5276', ms=13, zorder=5)
    ctext(ax, 3.0, 0.55, '相机', fontsize=10, color='#1A5276')

    # 3D Ellipsoid
    ell3d = Ellipse((3.0, 3.9), width=2.4, height=1.4, angle=18,
                    fc='#85C1E9', ec='#2874A6', lw=2.5, alpha=0.6, zorder=3)
    ax.add_patch(ell3d)
    arc3a = Arc((3.0, 3.9), 2.4, 0.48, angle=18,
                theta1=0, theta2=180, color='#2874A6', lw=1.5, ls='--')
    arc3b = Arc((3.0, 3.9), 2.4, 0.48, angle=18,
                theta1=180, theta2=360, color='#2874A6', lw=1.5)
    ax.add_patch(arc3a)
    ax.add_patch(arc3b)
    ax.plot(3.0, 3.9, 'o', color='#E74C3C', ms=7, zorder=6)
    ctext(ax, 4.05, 4.55, '3D 高斯\nΣ = R·S·Sᵀ·Rᵀ',
          fontsize=9.5, color='#1A5276', ha='left')

    # Projection rays from camera to ellipsoid
    for xe, ye in [(2.15, 3.55), (3.0, 3.9), (3.85, 3.55)]:
        ax.plot([3.0, xe], [0.9, ye], '--',
                color='#F39C12', lw=1.2, alpha=0.6, zorder=2)

    # ── Middle transition ───────────────────────────────────────────────────
    ax.annotate('', xy=(8.0, 3.9), xytext=(6.2, 3.9),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=3.5))
    ctext(ax, 7.1, 4.4, '投影变换', bold=True, fontsize=11, color='#2C3E50')

    jbox = FancyBboxPatch((6.15, 2.6), 1.9, 1.0,
                          boxstyle='round,pad=0.07',
                          fc='#FDEBD0', ec='#CA6F1E', lw=1.8)
    ax.add_patch(jbox)
    ctext(ax, 7.1, 3.1, '雅可比矩阵 J\n仿射近似', fontsize=9.5, color='#CA6F1E')
    ctext(ax, 7.1, 2.2,
          "Σ'₂D = J·W·Σ·Wᵀ·Jᵀ",
          fontsize=10, color='#7D3C98', style='italic')

    # ── Right panel: 2D screen ─────────────────────────────────────────────
    screen_bg = FancyBboxPatch((8.1, 0.5), 5.7, 6.5,
                               boxstyle='round,pad=0.12',
                               fc='#F9EBEA', ec='#C0392B', lw=2)
    ax.add_patch(screen_bg)
    ctext(ax, 11.0, 6.75, '二维图像平面', bold=True, fontsize=12, color='#C0392B')

    # Pixel grid
    screen_rect = FancyBboxPatch((8.7, 1.0), 4.6, 5.2,
                                 boxstyle='square,pad=0',
                                 fc='#D6EAF8', ec='#2874A6', lw=2)
    ax.add_patch(screen_rect)
    for gx in np.linspace(8.7, 13.3, 10):
        ax.plot([gx, gx], [1.0, 6.2], '-',
                color='#BDC3C7', lw=0.5, alpha=0.6)
    for gy in np.linspace(1.0, 6.2, 8):
        ax.plot([8.7, 13.3], [gy, gy], '-',
                color='#BDC3C7', lw=0.5, alpha=0.6)

    # 2D Ellipse
    ell2d = Ellipse((11.0, 3.7), width=3.2, height=1.5, angle=12,
                    fc='#AED6F1', ec='#1A5276', lw=2.5, alpha=0.75, zorder=4)
    ax.add_patch(ell2d)
    ax.plot(11.0, 3.7, 'o', color='#E74C3C', ms=8, zorder=6)

    ctext(ax, 11.0, 2.6, "μ'₂D （投影质心）", fontsize=10, color='#1A5276')
    ctext(ax, 12.3, 4.5, "α 不透明度\n按深度排序\nα-blending",
          fontsize=9, color='#7D3C98', ha='left')

    out = os.path.join(OUT_DIR, 'ch6_fig3.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FDFEFE')
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — ADC自适应密度控制
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5))
    fig.patch.set_facecolor('#FDFEFE')
    fig.suptitle('ADC 自适应密度控制（Adaptive Density Control）',
                 fontproperties=fp_bold, fontsize=16,
                 color='#1A5276', y=0.97)

    configs = [
        ('克隆（Clone）', '#27AE60', '#EAFAF1',
         '触发条件\n位置梯度 > τ_pos\n且高斯尺度 < 场景阈值'),
        ('分裂（Split）', '#2874A6', '#EBF5FB',
         '触发条件\n位置梯度 > τ_pos\n且高斯尺度 ≥ 场景阈值'),
        ('剪枝（Prune）', '#C0392B', '#FDEDEC',
         '触发条件\n不透明度 α < τ_α\n或高斯尺度超出范围'),
    ]

    for i, (ax, (title, col, bg, cond)) in enumerate(zip(axes, configs)):
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.8, 4.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(bg)
        ax.set_title(title, fontproperties=fp_bold,
                     fontsize=13, color=col, pad=12)

        if i == 0:  # Clone
            # Original small Gaussian
            e_orig = Ellipse((-0.1, -1.0), 1.1, 0.7, angle=0,
                             fc='#AED6F1', ec=col, lw=2.5, alpha=0.85)
            ax.add_patch(e_orig)
            ctext(ax, -0.1, -1.0, '小\n高斯', fontsize=10, color=col)

            # Arrow up
            ax.annotate('', xy=(-0.8, 0.8), xytext=(-0.8, -0.4),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
            ax.annotate('', xy=(0.8, 0.8), xytext=(0.8, -0.4),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.5))

            # Two identical clones
            for cx, cy in [(-1.1, 1.55), (1.1, 1.55)]:
                e_cl = Ellipse((cx, cy), 1.1, 0.7, angle=0,
                               fc='#D5F5E3', ec=col, lw=2, alpha=0.85)
                ax.add_patch(e_cl)
                ctext(ax, cx, cy, '克隆', fontsize=9.5, color=col)

            ctext(ax, 0, -2.1, '→ 复制为两个相同小高斯',
                  fontsize=10, color='#444', ha='center')
            ctext(ax, 0, -2.75, '（填充欠重建区域）',
                  fontsize=9, color='#666', ha='center')

        elif i == 1:  # Split
            # Original large Gaussian
            e_big = Ellipse((0, 0.3), 3.0, 1.6, angle=10,
                            fc='#FAD7A0', ec=col, lw=2.5, alpha=0.7)
            ax.add_patch(e_big)
            ctext(ax, 0, 0.3, '大高斯', fontsize=11, color=col, bold=True)

            # Split arrows
            ax.annotate('', xy=(-1.3, -1.7), xytext=(-0.6, -0.5),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
            ax.annotate('', xy=(1.3, -1.7), xytext=(0.6, -0.5),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.5))

            # Two smaller
            for sx in [-1.3, 1.3]:
                e_sm = Ellipse((sx, -2.3), 1.2, 0.75, angle=10,
                               fc='#D6EAF8', ec=col, lw=2, alpha=0.85)
                ax.add_patch(e_sm)
                ctext(ax, sx, -2.3, '小高斯', fontsize=9.5, color=col)

            ctext(ax, 0, 2.1, '→ 分裂为两个较小高斯',
                  fontsize=10, color='#444', ha='center')
            ctext(ax, 0, -3.35, '（精细化过大区域）',
                  fontsize=9, color='#666', ha='center')

        else:  # Prune
            # Three faint low-alpha Gaussians
            for px, py, alv in [(-1.5, 1.2, 0.12),
                                 (0.0, 0.0, 0.06),
                                 (1.5, 1.2, 0.09)]:
                e_faint = Ellipse((px, py), 1.2, 0.75, angle=0,
                                  fc='#D5D8DC', ec='#95A5A6', lw=1.5,
                                  alpha=0.30 + alv * 2)
                ax.add_patch(e_faint)
                ctext(ax, px, py, f'α≈{alv:.2f}', fontsize=9, color='#666')
                # X mark
                for sgn in [1, -1]:
                    ax.plot([px - 0.28 * sgn, px + 0.28 * sgn],
                            [py - 0.28, py + 0.28],
                            '-', color='#C0392B', lw=3.5)

            # Arrow down
            ax.annotate('', xy=(0, -1.6), xytext=(0, -0.5),
                        arrowprops=dict(arrowstyle='->', color=col, lw=3))

            del_box = FancyBboxPatch((-1.8, -2.9), 3.6, 1.1,
                                     boxstyle='round,pad=0.1',
                                     fc='#FDEDEC', ec=col, lw=2)
            ax.add_patch(del_box)
            ctext(ax, 0, -2.35, '已删除', bold=True,
                  fontsize=12, color=col)

            ctext(ax, 0, 2.5, '→ 透明度极低的基元被移除',
                  fontsize=10, color='#444', ha='center')
            ctext(ax, 0, -3.5, '（减少冗余，提升效率）',
                  fontsize=9, color='#666', ha='center')

        # Condition box at bottom
        c_box = FancyBboxPatch((-3.3, -4.0), 6.6, 1.5,
                               boxstyle='round,pad=0.1',
                               fc='#FDFEFE', ec=col, lw=1.8)
        # We draw this below the axes ylim — push up
        cond_y = -3.4
        cb = FancyBboxPatch((-3.3, cond_y - 0.7), 6.6, 1.3,
                            boxstyle='round,pad=0.1',
                            fc='#FDFEFE', ec=col, lw=1.5)
        ax.add_patch(cb)
        ax.set_ylim(-4.2, 4.0)
        ctext(ax, 0, cond_y - 0.05, cond, fontsize=8.5, color='#333')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, 'ch6_fig4.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FDFEFE')
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — 训练过程中高斯数量和loss变化
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#FDFEFE')
    ax1.set_facecolor('#F7F9FC')

    np.random.seed(42)
    iters = np.linspace(0, 30000, 600)

    # Loss curve
    loss_base = 0.20 * np.exp(-iters / 5500) + 0.013
    noise = 0.004 * np.random.randn(600) * np.exp(-iters / 9000)
    loss = np.clip(loss_base + noise, 0.009, None)
    kernel = np.ones(30) / 30
    loss_s = np.convolve(loss, kernel, mode='same')

    # Gaussian count curve
    phase1 = 60000 * (1 - np.exp(-iters / 4500))
    phase2 = 80000 * (1 - np.exp(-iters / 12000))
    dip = np.exp(-((iters - 20000) ** 2) / (2 * 8000 ** 2)) * 5000
    gauss_base = phase1 + phase2 - dip + 12000
    gauss_noise = 2500 * np.random.randn(600)
    gauss = np.clip(gauss_base + gauss_noise, 12000, None)
    gauss_s = np.convolve(gauss, kernel, mode='same')

    col_loss = '#E74C3C'
    col_gauss = '#2874A6'

    ln1, = ax1.plot(iters, loss_s, '-', color=col_loss, lw=2.5,
                    label='训练 Loss')
    ax1.set_xlabel('训练迭代次数', fontproperties=fp, fontsize=12)
    ax1.set_ylabel('Loss 值', fontproperties=fp, fontsize=12, color=col_loss)
    ax1.tick_params(axis='y', labelcolor=col_loss)
    ax1.set_xlim(0, 30000)
    ax1.set_ylim(0, 0.24)
    ax1.grid(True, alpha=0.3, ls='--')

    ax2 = ax1.twinx()
    ln2, = ax2.plot(iters, gauss_s / 1000, '--', color=col_gauss,
                    lw=2.5, label='高斯基元数量')
    ax2.set_ylabel('高斯基元数量（×10³）', fontproperties=fp,
                   fontsize=12, color=col_gauss)
    ax2.tick_params(axis='y', labelcolor=col_gauss)
    ax2.set_ylim(0, 220)

    # Milestone annotations
    milestones = [
        (500,   0.205, 'ADC 启动\n(500 步)'),
        (15000, 0.205, 'ADC 周期\n趋于稳定'),
        (30000, 0.065, '收敛'),
    ]
    for xv, yv, lbl in milestones:
        ax1.axvline(x=xv, color='#95A5A6', lw=1.2, ls=':')
        ax1.text(xv + 350, yv, lbl, fontproperties=fp,
                 fontsize=8.5, color='#555', va='top')

    lns = [ln1, ln2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, prop=fp, loc='upper right',
               fontsize=10, framealpha=0.92)

    ax1.set_title('训练过程：Loss 与高斯基元数量变化曲线',
                  fontproperties=fp_bold, fontsize=14,
                  color='#1A5276', pad=10)
    ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    ax1.set_xticklabels(['0', '5k', '10k', '15k', '20k', '25k', '30k'])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ch6_fig5.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FDFEFE')
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — L1 vs SSIM 损失函数对比
# ═══════════════════════════════════════════════════════════════════════════
def fig6():
    np.random.seed(7)

    # Create synthetic 32x32 colorful patch
    def make_sharp(n=32):
        img = np.zeros((n, n, 3))
        colors = [
            [0.88, 0.20, 0.18], [0.18, 0.68, 0.28], [0.18, 0.30, 0.88],
            [0.90, 0.78, 0.10], [0.58, 0.18, 0.80], [0.10, 0.78, 0.90],
            [0.90, 0.48, 0.10], [0.28, 0.90, 0.58],
        ]
        block = n // 4
        for r in range(n):
            for c in range(n):
                idx = (r // block) * 4 + (c // block)
                img[r, c] = colors[idx % len(colors)]
        return img

    sharp = make_sharp(32)
    blurry = np.clip(gaussian_filter(sharp, sigma=[3.0, 3.0, 0]), 0, 1)
    shifted = np.roll(sharp, 5, axis=1)     # structural shift

    # SSIM approximation map
    def ssim_map(a, b, win=5):
        g1, g2 = a.mean(2), b.mean(2)
        m1 = uniform_filter(g1, win)
        m2 = uniform_filter(g2, win)
        s1 = uniform_filter(g1**2, win) - m1**2
        s2 = uniform_filter(g2**2, win) - m2**2
        s12 = uniform_filter(g1 * g2, win) - m1 * m2
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*m1*m2 + C1) * (2*s12 + C2) /
                ((m1**2 + m2**2 + C1) * (s1 + s2 + C2)))
        return 1 - ssim

    l1_diff_blur   = np.abs(sharp - blurry).mean(2)
    l1_diff_shift  = np.abs(sharp - shifted).mean(2)
    ssim_diff_blur = ssim_map(sharp, blurry)
    ssim_diff_shift = ssim_map(sharp, shifted)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#FDFEFE')
    fig.suptitle('损失函数对比：L1 损失 vs SSIM 结构相似性损失',
                 fontproperties=fp_bold, fontsize=16, color='#1A5276', y=0.98)

    # Section backgrounds
    for x, col in [(0.01, '#2874A6'), (0.51, '#C0392B')]:
        bg = FancyBboxPatch((x, 0.03), 0.48, 0.92,
                            boxstyle='round,pad=0.01',
                            fc='#F8FBFF' if col == '#2874A6' else '#FEF6F5',
                            ec=col, lw=2.5,
                            transform=fig.transFigure, zorder=0)
        fig.add_artist(bg)

    fig.text(0.25, 0.93, 'L1 损失（像素级绝对误差）',
             fontproperties=fp_bold, fontsize=13, color='#2874A6', ha='center')
    fig.text(0.75, 0.93, 'SSIM 损失（块级结构感知）',
             fontproperties=fp_bold, fontsize=13, color='#C0392B', ha='center')

    # ─ L1 section ─
    # Row: reference | blurry prediction
    ax_ref_l1  = fig.add_axes([0.04, 0.58, 0.18, 0.28])
    ax_pred_l1 = fig.add_axes([0.27, 0.58, 0.18, 0.28])
    ax_diff_l1 = fig.add_axes([0.04, 0.20, 0.42, 0.30])

    ax_ref_l1.imshow(sharp, interpolation='nearest')
    ax_ref_l1.set_title('参考（清晰）', fontproperties=fp, fontsize=9.5,
                         color='#2874A6')
    ax_ref_l1.axis('off')

    ax_pred_l1.imshow(blurry, interpolation='bilinear')
    ax_pred_l1.set_title('预测（模糊）', fontproperties=fp, fontsize=9.5,
                          color='#2874A6')
    ax_pred_l1.axis('off')

    im1 = ax_diff_l1.imshow(l1_diff_blur, cmap='hot',
                              vmin=0, vmax=0.5, interpolation='nearest')
    ax_diff_l1.set_title(
        f'L1 差异热图  均值={l1_diff_blur.mean():.3f}',
        fontproperties=fp, fontsize=9.5, color='#2874A6')
    ax_diff_l1.axis('off')
    plt.colorbar(im1, ax=ax_diff_l1, fraction=0.046, pad=0.04)

    fig.text(0.25, 0.16,
             r'$\mathcal{L}_1 = \frac{1}{N}\sum_{i}|I^i_{ref} - I^i_{pred}|$',
             fontsize=12, color='#2874A6', ha='center')
    fig.text(0.25, 0.10,
             '逐像素独立计算；对高频噪声和模糊均敏感',
             fontproperties=fp, fontsize=9, color='#555', ha='center')

    # ─ SSIM section ─
    ax_ref_ss  = fig.add_axes([0.54, 0.58, 0.18, 0.28])
    ax_pred_ss = fig.add_axes([0.77, 0.58, 0.18, 0.28])
    ax_diff_ss = fig.add_axes([0.54, 0.20, 0.42, 0.30])

    ax_ref_ss.imshow(sharp, interpolation='nearest')
    ax_ref_ss.set_title('参考图像', fontproperties=fp, fontsize=9.5,
                         color='#C0392B')
    ax_ref_ss.axis('off')

    ax_pred_ss.imshow(shifted, interpolation='nearest')
    ax_pred_ss.set_title('结构偏移图像', fontproperties=fp, fontsize=9.5,
                          color='#C0392B')
    ax_pred_ss.axis('off')

    im2 = ax_diff_ss.imshow(ssim_diff_shift, cmap='RdYlGn_r',
                             vmin=0, vmax=1.0, interpolation='nearest')
    ax_diff_ss.set_title(
        f'SSIM 差异图（1-SSIM）  均值={ssim_diff_shift.mean():.3f}',
        fontproperties=fp, fontsize=9.5, color='#C0392B')
    ax_diff_ss.axis('off')
    plt.colorbar(im2, ax=ax_diff_ss, fraction=0.046, pad=0.04)

    fig.text(0.75, 0.16,
             r'$\mathcal{L}_{SSIM} = 1 - SSIM(I_{ref},\, I_{pred})$',
             fontsize=12, color='#C0392B', ha='center')
    fig.text(0.75, 0.10,
             '在滑动窗口内评估亮度/对比度/结构；对结构偏移高度敏感',
             fontproperties=fp, fontsize=9, color='#555', ha='center')

    # Combined loss
    fig.text(0.5, 0.04,
             '3DGS 总损失：  L = (1 − λ) · L₁  +  λ · L_SSIM    （λ = 0.2）',
             fontproperties=fp_bold, fontsize=12, ha='center',
             color='#1A5276',
             bbox=dict(fc='#EBF5FB', ec='#1A5276', lw=2,
                       boxstyle='round,pad=0.4'))

    out = os.path.join(OUT_DIR, 'ch6_fig6.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FDFEFE')
    plt.close()
    print(f'已保存 {out}')


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('生成第6章插图...')
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print('全部 6 幅图已保存完毕。')
