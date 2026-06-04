import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
import matplotlib.patheffects as pe
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图5.1 体渲染光线穿介质示意 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 3)
ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')

# 背景介质区域
medium_rect = mpatches.FancyBboxPatch((1.5, -0.6), 7.0, 2.2,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor='#95A5A6', facecolor='#EBF5FB', alpha=0.6)
ax.add_patch(medium_rect)
ax.text(5.0, 1.85, '体积介质', ha='center', fontsize=11,
        color='#2C3E50', fontweight='bold')

# 光线（渐变颜色表示强度衰减）
n_segments = 50
x_start, x_end = 0.2, 9.8
y_ray = 0.5
xs = np.linspace(x_start, x_end, n_segments + 1)

for i in range(n_segments):
    intensity = 1.0 - 0.75 * (i / n_segments)
    color = plt.cm.autumn(intensity)
    ax.plot([xs[i], xs[i+1]], [y_ray, y_ray],
            color=color, lw=6, solid_capstyle='butt', alpha=0.9)

# 入射光箭头标注
ax.annotate('', xy=(1.55, y_ray), xytext=(0.2, y_ray),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
ax.text(0.15, y_ray + 0.45, '入射光\n$I_0$', ha='center', fontsize=10,
        color='#E74C3C', fontweight='bold')

# 透射光箭头标注
ax.annotate('', xy=(9.8, y_ray), xytext=(8.5, y_ray),
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.5))
ax.text(9.85, y_ray + 0.45, '透射光\n$I(s)$', ha='center', fontsize=10,
        color='#E67E22', fontweight='bold')

# 粒子（体积介质中的散射/吸收粒子）
np.random.seed(42)
n_particles = 18
px = np.random.uniform(1.8, 8.3, n_particles)
py = np.random.uniform(-0.35, 1.35, n_particles)
on_ray = np.abs(py - y_ray) < 0.35

for i, (x, y) in enumerate(zip(px, py)):
    if on_ray[i]:
        circle = plt.Circle((x, y), 0.12, color='#2980B9', alpha=0.85, zorder=5)
        ax.add_patch(circle)
    else:
        circle = plt.Circle((x, y), 0.09, color='#7FB3D3', alpha=0.6, zorder=4)
        ax.add_patch(circle)

# 吸收标注
abs_x = 3.2
ax.annotate('吸收', xy=(abs_x, y_ray), xytext=(abs_x - 0.3, y_ray + 1.1),
            fontsize=9.5, color='#8E44AD',
            arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5),
            ha='center')

# 散射标注
sca_x = 5.8
ax.plot([sca_x, sca_x + 0.5], [y_ray, y_ray + 0.7], '--',
        color='#27AE60', lw=1.5, alpha=0.8)
ax.plot([sca_x, sca_x + 0.5], [y_ray, y_ray - 0.7], '--',
        color='#27AE60', lw=1.5, alpha=0.8)
ax.annotate('散射', xy=(sca_x + 0.5, y_ray + 0.7),
            xytext=(sca_x + 1.0, y_ray + 1.2),
            fontsize=9.5, color='#27AE60',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
            ha='center')

# 颜色条（光线强度图例）
sm = plt.cm.ScalarMappable(cmap=plt.cm.autumn,
                            norm=plt.Normalize(vmin=0.25, vmax=1.0))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.035, pad=0.02, aspect=30,
                    location='bottom')
cbar.set_label('光线强度（从左到右递减）', fontsize=10)
cbar.set_ticks([0.25, 0.625, 1.0])
cbar.set_ticklabels(['弱', '中', '强'])

ax.set_title('图5.1  体渲染：光线穿越介质示意', fontsize=13,
             fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('docs/pic/ch5_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch5_fig1.png saved")

# ── 图5.2 Alpha合成叠加过程 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_xlim(0, 8)
ax.set_ylim(-0.5, 9.5)
ax.axis('off')
fig.patch.set_facecolor('#FDFEFE')

layers = [
    (0.85, '#E74C3C', '红色层'),
    (0.60, '#27AE60', '绿色层'),
    (0.45, '#2980B9', '蓝色层'),
    (0.70, '#F39C12', '橙色层'),
]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

composite = np.array([1.0, 1.0, 1.0])
intermediate_colors = []

y_positions = [8.0, 6.0, 4.0, 2.0]
block_w = 3.5
block_h = 1.3

for idx, ((alpha, color, label), y) in enumerate(zip(layers, y_positions)):
    rgb = np.array(hex_to_rgb(color))
    composite = rgb * alpha + composite * (1 - alpha)
    intermediate_colors.append(composite.copy())

    rect = mpatches.FancyBboxPatch((1.0, y - block_h/2), block_w, block_h,
        boxstyle="round,pad=0.05", linewidth=2,
        edgecolor='#2C3E50',
        facecolor=color, alpha=alpha + 0.1)
    ax.add_patch(rect)

    ax.text(1.0 + block_w + 0.25, y, f'$\\alpha_{idx+1}$ = {alpha:.2f}',
            va='center', fontsize=12, color='#2C3E50', fontweight='bold')

    ax.text(1.0 + block_w/2, y, label, ha='center', va='center',
            fontsize=11, color='white',
            path_effects=[pe.withStroke(linewidth=2, foreground='#2C3E50')])

    comp_rect = mpatches.FancyBboxPatch((6.2, y - 0.4), 0.8, 0.8,
        boxstyle="round,pad=0.03", linewidth=1.5,
        edgecolor='#7F8C8D',
        facecolor=tuple(composite))
    ax.add_patch(comp_rect)
    ax.text(6.6, y - 0.65, f'合成{idx+1}', ha='center', fontsize=8,
            color='#7F8C8D')

    if idx < len(layers) - 1:
        ax.annotate('', xy=(2.75, y - block_h/2 - 0.08),
                    xytext=(2.75, y - block_h/2 - 0.55),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.0))
        ax.text(3.0, y - block_h/2 - 0.3, '继续合成', fontsize=8.5,
                color='#7F8C8D', va='center')

# 最终合成结果
final_y = 0.3
final_rgb = tuple(intermediate_colors[-1])
final_rect = mpatches.FancyBboxPatch((0.5, final_y - 0.55), 4.5, 1.1,
    boxstyle="round,pad=0.08", linewidth=2.5,
    edgecolor='#E74C3C', facecolor=final_rgb)
ax.add_patch(final_rect)
ax.text(2.75, final_y, '最终合成结果', ha='center', va='center',
        fontsize=12, color='white', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='#2C3E50')])
ax.text(5.2, final_y, 'Over合成公式:\n$C = \\sum_i c_i \\alpha_i \\prod_{j<i}(1-\\alpha_j)$',
        va='center', fontsize=9.5, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                  edgecolor='#2980B9', alpha=0.9))

ax.set_title('图5.2  Alpha 合成叠加过程（前向合成）', fontsize=13,
             fontweight='bold', pad=12)
ax.text(6.6, 9.2, '逐层\n预览', ha='center', fontsize=9,
        color='#7F8C8D', fontstyle='italic')

plt.tight_layout()
plt.savefig('docs/pic/ch5_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch5_fig2.png saved")

# ── 图5.3 NeRF vs 3DGS 渲染流程对比 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 8))
fig.patch.set_facecolor('#FAFAFA')

def draw_flow(ax, title, steps, colors, speed_notes, col_color):
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, len(steps) * 2.1 + 0.5)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    box_w, box_h = 4.0, 1.1
    cx = 3.0

    for i, (step, color, note) in enumerate(zip(steps, colors, speed_notes)):
        y = (len(steps) - 1 - i) * 2.0 + 0.5
        rect = mpatches.FancyBboxPatch((cx - box_w/2, y - box_h/2),
            box_w, box_h,
            boxstyle="round,pad=0.12", linewidth=2,
            edgecolor=col_color, facecolor=color, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, y, step, ha='center', va='center',
                fontsize=11, color='#2C3E50', fontweight='bold', zorder=4)
        if note:
            ax.text(cx + box_w/2 + 0.15, y, note,
                    va='center', fontsize=9, color='#E74C3C',
                    fontweight='bold')
        if i < len(steps) - 1:
            y_next = (len(steps) - 2 - i) * 2.0 + 0.5
            ax.annotate('', xy=(cx, y_next + box_h/2 + 0.08),
                        xytext=(cx, y - box_h/2 - 0.08),
                        arrowprops=dict(arrowstyle='->', color=col_color,
                                        lw=2.2, mutation_scale=18), zorder=3)

    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=col_color, pad=10)

nerf_steps = ['发射光线\n(Ray Casting)', '均匀/重要性采样\n(~192点/光线)',
              'MLP网络查询\n(颜色+密度)', '体渲染积分\n(Alpha合成)', '输出像素']
nerf_colors = ['#D6EAF8', '#AED6F1', '#7FB3D3', '#5499C7', '#2980B9']
nerf_notes = ['', '采样瓶颈', '慢（MLP推理）', '', '~30秒/帧']

gs_steps = ['3D 高斯基元\n(显式表达)', '投影到2D\n(EWA近似)',
            '按深度排序\n(Tile-based)', 'Alpha 合成\n(前向光栅化)', '输出像素']
gs_colors = ['#D5F5E3', '#A9DFBF', '#76D7A3', '#52BE80', '#27AE60']
gs_notes = ['', 'GPU并行', '高效排序', '光栅化', '~30 FPS']

draw_flow(axes[0], 'NeRF 渲染流程', nerf_steps, nerf_colors, nerf_notes, '#2980B9')
draw_flow(axes[1], '3DGS 渲染流程', gs_steps, gs_colors, gs_notes, '#27AE60')

fig.text(0.5, 0.02,
    '速度对比：3DGS（实时30+ FPS）vs NeRF（数秒/帧）— 提速约100倍',
    ha='center', fontsize=11, color='#8E44AD', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5EEF8',
              edgecolor='#8E44AD', alpha=0.9))

fig.suptitle('图5.3  NeRF vs 3DGS 渲染流程对比', fontsize=14,
             fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig('docs/pic/ch5_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch5_fig3.png saved")

# ── 图5.4 Tile-based 光栅化示意 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

n_tiles = 4
tile_size = 2.0

# 绘制 tile 网格背景
for i in range(n_tiles):
    for j in range(n_tiles):
        x0 = j * tile_size
        y0 = i * tile_size
        rect = mpatches.FancyBboxPatch((x0, y0), tile_size, tile_size,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor='#95A5A6', facecolor='#ECF0F1', alpha=0.5, zorder=1)
        ax.add_patch(rect)
        ax.text(x0 + tile_size/2, y0 + tile_size/2,
                f'T({j},{i})', ha='center', va='center',
                fontsize=8.5, color='#7F8C8D', alpha=0.7, zorder=2)

# 定义2D高斯椭圆
gaussians = [
    (2.2, 5.8, 2.0, 1.0, 25,  '#E74C3C', 'G1'),
    (5.5, 5.0, 1.5, 0.7, -15, '#2980B9', 'G2'),
    (3.5, 2.5, 2.2, 0.8, 10,  '#27AE60', 'G3'),
    (6.5, 2.0, 1.2, 1.4, 45,  '#F39C12', 'G4'),
    (1.2, 1.5, 1.0, 0.6, 0,   '#8E44AD', 'G5'),
]

# 高亮各高斯覆盖的tiles
highlight_alpha = 0.18
for (cx, cy, rx, ry, ang, color, label) in gaussians:
    ang_rad = np.radians(ang)
    cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
    for i in range(n_tiles):
        for j in range(n_tiles):
            tx0 = j * tile_size
            ty0 = i * tile_size
            tile_cx = tx0 + tile_size / 2
            tile_cy = ty0 + tile_size / 2
            dx = tile_cx - cx
            dy = tile_cy - cy
            dx_rot = dx * cos_a + dy * sin_a
            dy_rot = -dx * sin_a + dy * cos_a
            expand = tile_size * 0.8
            if (dx_rot / (rx + expand))**2 + (dy_rot / (ry + expand))**2 < 1.0:
                hlight = mpatches.FancyBboxPatch((tx0, ty0), tile_size, tile_size,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor=color, alpha=highlight_alpha, zorder=2)
                ax.add_patch(hlight)

# 绘制高斯椭圆
for (cx, cy, rx, ry, ang, color, label) in gaussians:
    ell = Ellipse((cx, cy), width=rx*2, height=ry*2, angle=ang,
                  edgecolor=color, facecolor=color,
                  linewidth=2.5, alpha=0.35, zorder=4)
    ax.add_patch(ell)
    ell2 = Ellipse((cx, cy), width=rx*2, height=ry*2, angle=ang,
                   edgecolor=color, facecolor='none',
                   linewidth=2.5, alpha=0.95, zorder=5)
    ax.add_patch(ell2)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=11, color=color, fontweight='bold', zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# 重绘清晰网格线
for i in range(n_tiles + 1):
    ax.axhline(i * tile_size, color='#7F8C8D', lw=1.5, alpha=0.8, zorder=7)
    ax.axvline(i * tile_size, color='#7F8C8D', lw=1.5, alpha=0.8, zorder=7)

# 标注 tile 宽度
ax.annotate('', xy=(2.0, -0.35), xytext=(0.0, -0.35),
            arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=1.8))
ax.text(1.0, -0.45, 'tile 宽度', ha='center', va='top', fontsize=9,
        color='#2C3E50')

# 图例
legend_elems = []
for (_, _, _, _, _, color, label) in gaussians:
    patch = mpatches.Patch(facecolor=color, alpha=0.5,
                           edgecolor=color, label=f'{label} 覆盖范围')
    legend_elems.append(patch)
ax.legend(handles=legend_elems, loc='upper right',
          fontsize=9, framealpha=0.9,
          bbox_to_anchor=(1.0, 1.0))

# 并行处理标注
ax.text(4.0, 8.2,
    '每个 Tile 独立并行处理，GPU 高吞吐',
    ha='center', fontsize=10.5, color='#8E44AD', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5EEF8',
              edgecolor='#8E44AD', alpha=0.9))

ax.set_title('图5.4  Tile-based 光栅化：4x4 Tile 网格与高斯覆盖范围',
             fontsize=13, fontweight='bold', pad=14)

plt.tight_layout()
plt.savefig('docs/pic/ch5_fig4.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch5_fig4.png saved")
print("ch5 all done")
