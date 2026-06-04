import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.font_manager import FontProperties
import os
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)


# ─────────────────────────────────────────────
# Figure 1: 双目视差示意图
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlim(-1, 6)
ax1.set_ylim(-1, 5.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('双目视差示意图', fontsize=16, fontweight='bold', pad=14)

# Eye positions
left_eye_pos  = (1.0, 0.5)
right_eye_pos = (4.0, 0.5)

# Target (triangle) at top centre
target_x, target_y = 2.5, 4.2

# Draw eyes (circles)
eye_radius = 0.25
for pos, label in [(left_eye_pos, '左眼'), (right_eye_pos, '右眼')]:
    circle = plt.Circle(pos, eye_radius, color='#4C72B0', zorder=5)
    ax1.add_patch(circle)
    ax1.text(pos[0], pos[1] - 0.55, label,
             ha='center', va='top', fontsize=11, color='#4C72B0', fontweight='bold')

# Draw target triangle
triangle = plt.Polygon(
    [[target_x - 0.4, target_y - 0.5],
     [target_x + 0.4, target_y - 0.5],
     [target_x,       target_y + 0.1]],
    closed=True, color='#DD8452', zorder=5)
ax1.add_patch(triangle)
ax1.text(target_x + 0.6, target_y - 0.2, '观察目标',
         ha='left', va='center', fontsize=11, color='#DD8452', fontweight='bold')

# Draw sight lines
for eye_pos, ls in [(left_eye_pos, '--'), (right_eye_pos, '-.')]:
    ax1.annotate('', xy=(target_x, target_y - 0.5),
                 xytext=(eye_pos[0], eye_pos[1] + eye_radius),
                 arrowprops=dict(arrowstyle='->', color='#555555',
                                 lw=1.8, linestyle=ls))

# Baseline (distance between eyes)
ax1.annotate('', xy=(right_eye_pos[0], left_eye_pos[1] - 0.15),
             xytext=(left_eye_pos[0], left_eye_pos[1] - 0.15),
             arrowprops=dict(arrowstyle='<->', color='#229954', lw=1.8))
ax1.text((left_eye_pos[0] + right_eye_pos[0]) / 2, left_eye_pos[1] - 0.38,
         '眼距（基线）', ha='center', va='top', fontsize=10, color='#229954')

# Depth arrow (vertical)
ax1.annotate('', xy=(target_x + 1.5, target_y - 0.5),
             xytext=(target_x + 1.5, left_eye_pos[1] + eye_radius),
             arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=1.8))
ax1.text(target_x + 1.75, (target_y - 0.5 + left_eye_pos[1] + eye_radius) / 2,
         '深度', ha='left', va='center', fontsize=11, color='#C0392B', fontweight='bold')

# Parallax angle annotation
ax1.annotate('', xy=(left_eye_pos[0] + 0.3, left_eye_pos[1] + 0.5),
             xytext=(right_eye_pos[0] - 0.3, right_eye_pos[1] + 0.5),
             arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=1.5,
                             connectionstyle='arc3,rad=-0.4'))
ax1.text((left_eye_pos[0] + right_eye_pos[0]) / 2, left_eye_pos[1] + 1.05,
         '视差角', ha='center', va='bottom', fontsize=11, color='#8E44AD', fontweight='bold')

fig1.tight_layout()
fig1.savefig('docs/pic/ch1_fig1.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print('Figure 1 saved: docs/pic/ch1_fig1.png')


# ─────────────────────────────────────────────
# Figure 2: 三大技术路线对比横条图
# ─────────────────────────────────────────────
methods = ['MVS 传统方法', 'NeRF 神经隐式', '3DGS 显式高斯']
metrics = ['训练速度', '渲染速度', '重建质量', '内存占用']

# Scores (0-10): rows = methods, cols = metrics
scores = np.array([
    [8, 9, 5, 8],   # MVS
    [2, 2, 8, 5],   # NeRF
    [6, 9, 8, 4],   # 3DGS
])

n_metrics = len(metrics)
x         = np.arange(n_metrics)
bar_h     = 0.22
offsets   = np.array([-1, 0, 1]) * bar_h

colors = ['#4C72B0', '#DD8452', '#55A868']

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2.set_title('三大技术路线综合对比', fontsize=16, fontweight='bold', pad=14)

for i, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
    bars = ax2.barh(x + offset, scores[i], height=bar_h * 0.9,
                    color=color, alpha=0.85, label=method)
    for bar, score in zip(bars, scores[i]):
        ax2.text(score + 0.15, bar.get_y() + bar.get_height() / 2,
                 str(score), va='center', fontsize=9, color=color, fontweight='bold')

ax2.set_yticks(x)
ax2.set_yticklabels(metrics, fontsize=12)
ax2.set_xlabel('评分（0 – 10 分）', fontsize=11)
ax2.set_xlim(0, 11.5)
ax2.axvline(x=5, color='grey', linestyle=':', linewidth=1, alpha=0.6)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.grid(axis='x', linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig('docs/pic/ch1_fig2.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print('Figure 2 saved: docs/pic/ch1_fig2.png')


# ─────────────────────────────────────────────
# Figure 3: 章节依赖关系图
# ─────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 6))
ax3.set_xlim(0, 11)
ax3.set_ylim(0, 5)
ax3.axis('off')
ax3.set_title('本教程章节依赖关系', fontsize=16, fontweight='bold', pad=14)

# Chapter definitions: (id, label, x, y)
chapters = [
    (1,  '第1章\n概述',           1.0,  3.5),
    (2,  '第2章\n基础数学',        2.5,  3.5),
    (3,  '第3章\n相机模型',        4.0,  4.5),
    (4,  '第4章\n点云处理',        4.0,  3.5),
    (5,  '第5章\nNeRF基础',       5.5,  4.5),
    (6,  '第6章\n高斯表示',        5.5,  3.5),
    (7,  '第7章\n3DGS核心',       7.0,  3.5),
    (8,  '第8章\n训练优化',        8.5,  4.2),
    (9,  '第9章\n应用场景',        8.5,  2.8),
    (10, '第10章\n进阶专题',       10.0, 3.5),
]

# Box styling
box_w, box_h = 1.1, 0.65
node_color   = '#D6EAF8'
border_color = '#2980B9'

pos = {cid: (cx, cy) for cid, _, cx, cy in chapters}


def draw_box(ax, cx, cy, label, highlight=False):
    fc = '#FDEBD0' if highlight else node_color
    ec = '#E67E22'  if highlight else border_color
    rect = mpatches.FancyBboxPatch(
        (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
        boxstyle='round,pad=0.08', facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=8.5, zorder=4, linespacing=1.4)


for cid, label, cx, cy in chapters:
    highlight = cid in (7, 10)
    draw_box(ax3, cx, cy, label, highlight=highlight)

# Main dependency arrows
main_edges = [
    (1, 2), (2, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (6, 7), (7, 8), (7, 9), (8, 10), (9, 10),
]


def arrow(ax, start, end, color='#2C3E50', lw=1.5, rad=0.0, ls='-'):
    sx, sy = pos[start]
    ex, ey = pos[end]
    dx, dy = ex - sx, ey - sy
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    sx2 = sx + ux * box_w / 2
    sy2 = sy + uy * box_h / 2
    ex2 = ex - ux * box_w / 2
    ey2 = ey - uy * box_h / 2
    ax.annotate('', xy=(ex2, ey2), xytext=(sx2, sy2),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)


for s, e in main_edges:
    arrow(ax3, s, e, color='#2C3E50', lw=1.6)

# Skip paths (dashed) — 有数学基础可跳过3/4章
skip_edges = [
    (2, 6, '有数学基础\n可跳过3-4章',  0.25, '#C0392B'),
    (2, 5, '',                          -0.22, '#C0392B'),
]
for s, e, note, rad, col in skip_edges:
    arrow(ax3, s, e, color=col, lw=1.4, rad=rad, ls='dashed')
    if note:
        sx, sy = pos[s]
        ex, ey = pos[e]
        mx = (sx + ex) / 2 + 0.2
        my = (sy + ey) / 2 + 0.55
        ax3.text(mx, my, note, ha='center', va='bottom',
                 fontsize=8, color=col, style='italic',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC',
                           edgecolor=col, alpha=0.85))

# Legend
legend_items = [
    mpatches.Patch(facecolor='#D6EAF8', edgecolor='#2980B9', label='普通章节'),
    mpatches.Patch(facecolor='#FDEBD0', edgecolor='#E67E22', label='核心章节'),
    mpatches.Patch(facecolor='white',   edgecolor='#2C3E50', label='必学路径'),
    mpatches.Patch(facecolor='white',   edgecolor='#C0392B', label='可跳跃路径（虚线）'),
]
ax3.legend(handles=legend_items, loc='lower left', fontsize=8.5,
           framealpha=0.9, ncol=2)

fig3.tight_layout()
fig3.savefig('docs/pic/ch1_fig3.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print('Figure 3 saved: docs/pic/ch1_fig3.png')

print('ch1 all done')
