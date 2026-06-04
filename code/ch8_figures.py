import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图8.1 拍摄轨迹建议示意（俯视图）────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.axis('off')

# 背景浅灰网格
for v in np.arange(-5, 6, 1):
    ax.axhline(v, color='#EEEEEE', lw=0.5, zorder=0)
    ax.axvline(v, color='#EEEEEE', lw=0.5, zorder=0)

# 被拍摄物体（中央正方形）
obj = plt.Polygon([[-0.65, -0.65], [0.65, -0.65], [0.65, 0.65], [-0.65, 0.65]],
                  closed=True, facecolor='#D4E6F1', edgecolor='#2980B9', lw=2.5, zorder=3)
ax.add_patch(obj)
ax.text(0, 0, '被拍摄\n物体', ha='center', va='center', fontsize=11,
        color='#1A5276', fontweight='bold', zorder=4)

# 外圈路径（大范围拍摄）
outer_r = 3.8
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(outer_r * np.cos(theta), outer_r * np.sin(theta),
        color='#E74C3C', lw=2.0, ls='--', zorder=2)

# 内圈路径（近距离拍摄）
inner_r = 2.2
ax.plot(inner_r * np.cos(theta), inner_r * np.sin(theta),
        color='#27AE60', lw=2.0, ls='--', zorder=2)


def draw_camera(ax, cx, cy, angle_rad, color, size=0.30):
    """小三角形表示相机，尖端朝向圆心"""
    tip = np.array([cx, cy]) - size * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    left = np.array([cx, cy]) + size * 0.6 * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    right = np.array([cx, cy]) - size * 0.6 * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    tri = plt.Polygon([tip, left, right], closed=True,
                      facecolor=color, edgecolor='white', lw=1.0, zorder=5)
    ax.add_patch(tri)


# 外圈12个相机
n_outer = 12
for i in range(n_outer):
    ang = 2 * np.pi * i / n_outer
    draw_camera(ax, outer_r * np.cos(ang), outer_r * np.sin(ang), ang, '#E74C3C', size=0.30)

# 内圈8个相机（错开半步角度）
n_inner = 8
for i in range(n_inner):
    ang = 2 * np.pi * i / n_inner + np.pi / n_inner
    draw_camera(ax, inner_r * np.cos(ang), inner_r * np.sin(ang), ang, '#27AE60', size=0.28)

# 圈标签
ax.text(outer_r + 0.25, 0.4, '外圈\n大范围拍摄', fontsize=10,
        color='#C0392B', va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C', alpha=0.88))
ax.text(inner_r + 0.2, -1.8, '内圈\n近距离拍摄', fontsize=10,
        color='#1E8449', va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', edgecolor='#27AE60', alpha=0.88))

# 顺时针方向示意箭头
for ang_deg in [45, 135, 225, 315]:
    ang = np.radians(ang_deg)
    r_mid = (outer_r + inner_r) / 2
    ax.annotate('',
        xy=(r_mid * np.cos(ang + 0.28), r_mid * np.sin(ang + 0.28)),
        xytext=(r_mid * np.cos(ang - 0.28), r_mid * np.sin(ang - 0.28)),
        arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.2), zorder=1)

# 拍摄建议文字框
notes = ['建议重叠率 > 60%', '不同高度多层拍摄', '避免纯旋转运动', '保持匀速稳定移动']
note_x, note_y = -5.3, -2.8
ax.add_patch(FancyBboxPatch((note_x - 0.1, note_y - 1.7), 3.3, 2.0,
             boxstyle='round,pad=0.15', facecolor='#FEF9E7',
             edgecolor='#F39C12', lw=1.5, zorder=3))
ax.text(note_x + 1.5, note_y + 0.2, '拍摄建议', fontsize=11,
        color='#B7770D', fontweight='bold', ha='center', zorder=4)
for j, note in enumerate(notes):
    ax.text(note_x + 0.1, note_y - 0.38 - j * 0.4, f'• {note}',
            fontsize=9.5, color='#5D4037', zorder=4)

ax.set_title('图8.1  拍摄轨迹建议示意（俯视图）', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('docs/pic/ch8_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch8_fig1.png saved")

# ── 图8.2 数据集对比表格────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.axis('off')

col_labels = ['数据集名称', '场景类型', '图像数量', '分辨率', '主要用途']
rows_data = [
    ['Tanks & Temples',  '室外大场景 / 室内',  '151–1,000+',  '1920×1080',   '重建精度评测'],
    ['Mip-NeRF 360',     '无界室外 / 室内',    '100–300',     '2268×1512',   'NeRF / 3DGS对比'],
    ['Deep Blending',    '室内广角场景',        '约200',       '1920×1080',   '视图合成测试'],
    ['DTU',              '物体级受控场景',      '49 / 64',     '1600×1200',   '多视图重建'],
    ['KITTI-360',        '自动驾驶街景',        '数万帧',      '1408×376',    '大规模场景理解'],
]

cell_colors = []
for i in range(len(rows_data)):
    base = '#F2F3F4' if i % 2 == 0 else '#FFFFFF'
    cell_colors.append([base] * len(col_labels))

table = ax.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    cellColours=cell_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

# 表头样式
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=11.5)
    cell.set_edgecolor('#AAAAAA')

# 数据行边框 + 第一列加粗
for (row, col), cell in table.get_celld().items():
    if row > 0:
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.8)
        if col == 0:
            cell.set_text_props(fontweight='bold', color='#1A5276')

ax.set_title('图8.2  常用3DGS基准数据集对比', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('docs/pic/ch8_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch8_fig2.png saved")

# ── 图8.3 训练显存占用 vs 高斯数量────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

# 高斯数量（万）和对应显存（GB）— 贴近实测的估算值
gaussian_counts_w = np.array([10, 50, 100, 150, 200, 300, 400, 500, 600])  # 万
vram_usage = np.array([2.1, 3.4, 5.2, 6.8, 8.5, 11.8, 15.2, 18.9, 22.7])  # GB

ax.plot(gaussian_counts_w * 1e4, vram_usage, 'o-',
        color='#2980B9', lw=2.5, markersize=7, markerfacecolor='white',
        markeredgewidth=2.0, label='显存占用（估算）', zorder=4)

# 显存限制水平线
limits = [
    (8,  '#E74C3C', '8 GB（入门级 GPU）'),
    (16, '#E67E22', '16 GB（消费级 GPU）'),
    (24, '#27AE60', '24 GB（专业级 GPU）'),
]
for vram_lim, color, label in limits:
    ax.axhline(vram_lim, color=color, lw=1.8, ls='--', alpha=0.85, zorder=3)
    ax.text(5.8e6, vram_lim + 0.45, label, fontsize=10, color=color, ha='right', va='bottom')

# 可用范围着色
ax.fill_between(gaussian_counts_w * 1e4, 0, vram_usage,
                where=(vram_usage <= 8), color='#AED6F1', alpha=0.40,
                label='8 GB 可用范围', zorder=2)
ax.fill_between(gaussian_counts_w * 1e4, 0, vram_usage,
                where=(vram_usage > 8) & (vram_usage <= 16),
                color='#A9DFBF', alpha=0.40, label='8–16 GB 范围', zorder=2)

# 关键点标注
for gw, gv, note in [(100, 5.2, '~100万高斯\n5.2 GB'),
                     (300, 11.8, '~300万高斯\n11.8 GB'),
                     (500, 18.9, '~500万高斯\n18.9 GB')]:
    ax.annotate(note,
        xy=(gw * 1e4, gv),
        xytext=(gw * 1e4 + 28000, gv + 2.2),
        fontsize=9.5, color='#333',
        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                  edgecolor='#BBBBBB', alpha=0.9))

ax.set_xlabel('高斯数量（个）', fontsize=12)
ax.set_ylabel('显存占用（GB）', fontsize=12)
ax.set_title('图8.3  训练显存占用 vs 高斯数量', fontsize=14, fontweight='bold')
ax.set_xlim(0, 6.5e6)
ax.set_ylim(0, 26)
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x / 1e4)}万'))
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, ls='--')
plt.tight_layout()
plt.savefig('docs/pic/ch8_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch8_fig3.png saved")

# ── 图8.4 PSNR随训练迭代的变化────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

iters = np.array([0, 500, 1000, 2000, 3000, 5000, 7000,
                  10000, 15000, 20000, 25000, 30000])

# 室内场景（收敛更高）
indoor_psnr  = np.array([8.5, 18.2, 22.5, 25.8, 27.3, 28.9, 30.1,
                          31.2, 32.1, 32.8, 33.2, 33.5])
# 室外场景（收敛略低）
outdoor_psnr = np.array([7.8, 16.5, 20.1, 23.0, 24.5, 25.8, 26.9,
                          27.7, 28.4, 28.8, 29.1, 29.3])

ax.plot(iters, indoor_psnr, 'o-', color='#2980B9', lw=2.5, markersize=6,
        markerfacecolor='white', markeredgewidth=2.0, label='室内场景（如室内房间）', zorder=4)
ax.plot(iters, outdoor_psnr, 's-', color='#E74C3C', lw=2.5, markersize=6,
        markerfacecolor='white', markeredgewidth=2.0, label='室外场景（如花园）', zorder=4)

# 7000迭代检查点
ax.axvline(7000, color='#8E44AD', lw=1.8, ls='--', alpha=0.85, zorder=3)
ax.text(7300, 11.5, '7000步\n检查点', fontsize=10, color='#8E44AD', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8',
                  edgecolor='#8E44AD', alpha=0.9))

# 30000迭代检查点
ax.axvline(30000, color='#16A085', lw=1.8, ls='--', alpha=0.85, zorder=3)
ax.text(29700, 11.5, '30000步\n最终检查点', fontsize=10, color='#16A085',
        va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5',
                  edgecolor='#16A085', alpha=0.9))

# 7000步时的PSNR值标注
ax.annotate(f'室内 {indoor_psnr[6]:.1f} dB',
    xy=(7000, indoor_psnr[6]),
    xytext=(3000, indoor_psnr[6] + 1.4),
    fontsize=9.5, color='#2980B9',
    arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.2))
ax.annotate(f'室外 {outdoor_psnr[6]:.1f} dB',
    xy=(7000, outdoor_psnr[6]),
    xytext=(2500, outdoor_psnr[6] - 2.2),
    fontsize=9.5, color='#E74C3C',
    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))

# 30000步时的PSNR值标注
ax.annotate(f'室内 {indoor_psnr[-1]:.1f} dB',
    xy=(30000, indoor_psnr[-1]),
    xytext=(26500, indoor_psnr[-1] - 2.2),
    fontsize=9.5, color='#2980B9',
    arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.2))
ax.annotate(f'室外 {outdoor_psnr[-1]:.1f} dB',
    xy=(30000, outdoor_psnr[-1]),
    xytext=(26500, outdoor_psnr[-1] + 1.2),
    fontsize=9.5, color='#E74C3C',
    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))

ax.set_xlabel('训练迭代次数', fontsize=12)
ax.set_ylabel('PSNR（dB）', fontsize=12)
ax.set_title('图8.4  PSNR 随训练迭代的变化', fontsize=14, fontweight='bold')
ax.set_xlim(-500, 31500)
ax.set_ylim(6, 36)
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x > 0 else '0'))
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, ls='--')
plt.tight_layout()
plt.savefig('docs/pic/ch8_fig4.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch8_fig4.png saved")

print("ch8 all done")
