import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib.transforms as transforms
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图9.1 3DGS改进方向全景思维导图 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)

# 中心节点
cx, cy = 7.0, 5.5
center_box = FancyBboxPatch((cx - 1.5, cy - 0.65), 3.0, 1.3,
                             boxstyle='round,pad=0.3',
                             facecolor='#FF6B35', edgecolor='#C0392B', lw=2.5)
ax.add_patch(center_box)
ax.text(cx, cy, '3DGS (2023)', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

# 8个方向：(角度, 标签行1, 子节点列表, 颜色)
directions = [
    (90,  '压缩与效率',   ['Scaffold-GS', 'Mini-Splatting', 'Compact-3DGS'], '#4C72B0'),
    (45,  '几何重建',     ['2DGS', 'GOF', 'SuGaR'],                          '#2ECC71'),
    (0,   '动态场景',     ['4D Gaussians', 'Deformable 3DGS', 'SC-GS'],      '#E74C3C'),
    (315, '大场景',       ['VastGaussian', 'CityGaussian', 'Octree-GS'],     '#9B59B6'),
    (270, '生成与编辑',   ['GaussianDreamer', 'DreamGaussian', 'GaussianEditor'], '#F39C12'),
    (225, '语义理解',     ['Gaussian Grouping', 'Feature 3DGS', 'LangSplat'], '#1ABC9C'),
    (180, '重光照',       ['Relightable 3DGS', 'GS-IR', 'GaussianShader'],   '#E67E22'),
    (135, 'SLAM',         ['3DGS-SLAM', 'SplaTAM', 'MonoGS'],                '#8E44AD'),
]

radius_main = 3.6   # 中心到主节点距离
radius_child = 1.4  # 主节点到子节点偏移

for angle_deg, label, children, color in directions:
    rad = np.radians(angle_deg)
    mx = cx + radius_main * np.cos(rad)
    my = cy + radius_main * np.sin(rad)

    # 连线：中心 → 主节点
    ax.annotate('', xy=(mx, my), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0, alpha=0.7))

    # 主节点（圆形）
    circle = plt.Circle((mx, my), 0.62, facecolor=color, edgecolor='white',
                         lw=1.8, zorder=5, alpha=0.9)
    ax.add_patch(circle)
    ax.text(mx, my, label, ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='white', zorder=6)

    # 子节点：在主节点周围展开
    n = len(children)
    spread = 35  # 子节点展开角度范围（度）
    child_angles = np.linspace(angle_deg - spread, angle_deg + spread, n)
    for i, (child_label, child_angle) in enumerate(zip(children, child_angles)):
        crad = np.radians(child_angle)
        child_radius = radius_main + radius_child
        sx = cx + child_radius * np.cos(crad)
        sy = cy + child_radius * np.sin(crad)

        # 连线：主节点 → 子节点
        ax.plot([mx, sx], [my, sy], '-', color=color, lw=1.2, alpha=0.5, zorder=3)

        # 子节点（小圆角矩形）
        tw = 1.45
        th = 0.38
        sub_box = FancyBboxPatch((sx - tw / 2, sy - th / 2), tw, th,
                                  boxstyle='round,pad=0.08',
                                  facecolor=color, alpha=0.2,
                                  edgecolor=color, lw=1.2, zorder=4)
        ax.add_patch(sub_box)
        ax.text(sx, sy, child_label, ha='center', va='center',
                fontsize=7.5, color='#222', zorder=5)

ax.set_title('图9.1  3DGS 改进方向全景思维导图（截至 2026 年中）',
             fontsize=14, pad=12, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/pic/ch9_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch9_fig1.png saved")


# ── 图9.2 2DGS vs 3DGS 几何表示对比 ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('#FAFAFA')

# ---- 左子图：3DGS（椭球体） ----
ax3 = axes[0]
ax3.set_xlim(-3, 3)
ax3.set_ylim(-1.5, 3.8)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('3DGS — 三维高斯（体积表示）', fontsize=12, fontweight='bold',
              color='#2C3E50', pad=8)

# 绘制一段简单曲面（由多个椭球堆叠近似）
np.random.seed(7)
surface_x = np.linspace(-2.4, 2.4, 7)
surface_y_base = 0.5 * np.sin(surface_x * 0.7) + 0.2

ellipsoid_params = []  # (cx, cy, rx, ry, angle)
for i, (bx, by) in enumerate(zip(surface_x, surface_y_base)):
    rx = 0.55 + 0.08 * np.sin(i)
    ry = 0.38 + 0.05 * np.cos(i)
    angle = 12 * np.sin(i * 0.9)
    ellipsoid_params.append((bx, by, rx, ry, angle))

colors_3dgs = plt.cm.Blues(np.linspace(0.4, 0.85, len(ellipsoid_params)))

for idx, (bx, by, rx, ry, ang) in enumerate(ellipsoid_params):
    # 半透明体积感椭球
    e_outer = Ellipse((bx, by), rx * 2.2, ry * 2.2, angle=ang,
                       facecolor=colors_3dgs[idx], edgecolor='#2980B9',
                       alpha=0.35, lw=1.2, zorder=2)
    e_inner = Ellipse((bx, by), rx * 1.0, ry * 1.0, angle=ang,
                       facecolor=colors_3dgs[idx], edgecolor='#1A6EA8',
                       alpha=0.75, lw=0.8, zorder=3)
    ax3.add_patch(e_outer)
    ax3.add_patch(e_inner)
    ax3.plot(bx, by, 'o', color='#1A5276', ms=2.5, zorder=4)

# 注释
ax3.annotate('三维高斯体\n（具有体积）', xy=(-1.8, 0.6),
             xytext=(-2.7, 2.2), fontsize=9, color='#1A5276',
             arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.3),
             bbox=dict(facecolor='#D6EAF8', edgecolor='#2980B9',
                       boxstyle='round,pad=0.3', alpha=0.85))
ax3.annotate('高斯间存在\n体积重叠', xy=(0.5, 0.15),
             xytext=(1.2, 2.3), fontsize=9, color='#922B21',
             arrowprops=dict(arrowstyle='->', color='#922B21', lw=1.3),
             bbox=dict(facecolor='#FADBD8', edgecolor='#E74C3C',
                       boxstyle='round,pad=0.3', alpha=0.85))

ax3.text(0, -1.1, '3D高斯（体积表示）\n体积性强，表面不精确',
         ha='center', va='bottom', fontsize=10, color='#2C3E50',
         bbox=dict(facecolor='#EBF5FB', edgecolor='#85C1E9',
                   boxstyle='round,pad=0.4', alpha=0.9))

# ---- 右子图：2DGS（椭圆片/盘片） ----
ax2 = axes[1]
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1.5, 3.8)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('2DGS — 二维高斯片（表面表示）', fontsize=12, fontweight='bold',
              color='#2C3E50', pad=8)

colors_2dgs = plt.cm.Greens(np.linspace(0.45, 0.88, 7))

for idx, (bx, by, rx, ry, ang) in enumerate(ellipsoid_params):
    # 扁平椭圆片：ry很小表示薄片
    ry_flat = 0.10
    # 主面（贴合曲面法向量倾斜）
    e_flat = Ellipse((bx, by), rx * 2.0, ry_flat * 2.0,
                      angle=ang + 5,
                      facecolor=colors_2dgs[idx], edgecolor='#1E8449',
                      alpha=0.85, lw=1.5, zorder=3)
    ax2.add_patch(e_flat)
    # 高亮边缘
    e_edge = Ellipse((bx, by), rx * 2.0, ry_flat * 2.0,
                      angle=ang + 5,
                      facecolor='none', edgecolor='#27AE60',
                      alpha=1.0, lw=2.0, zorder=4)
    ax2.add_patch(e_edge)

# 绘制曲面轮廓线（突显贴合性）
smooth_x = np.linspace(-2.5, 2.5, 200)
smooth_y = 0.5 * np.sin(smooth_x * 0.7) + 0.2
ax2.plot(smooth_x, smooth_y, '--', color='#117A65', lw=1.5,
         alpha=0.55, label='理想表面', zorder=2)

ax2.annotate('二维高斯片\n（贴合表面）', xy=(-1.8, 0.5),
             xytext=(-2.7, 2.2), fontsize=9, color='#1E8449',
             arrowprops=dict(arrowstyle='->', color='#1E8449', lw=1.3),
             bbox=dict(facecolor='#D5F5E3', edgecolor='#27AE60',
                       boxstyle='round,pad=0.3', alpha=0.85))
ax2.annotate('片状结构\n无体积冗余', xy=(0.5, 0.15),
             xytext=(1.2, 2.3), fontsize=9, color='#1E8449',
             arrowprops=dict(arrowstyle='->', color='#1E8449', lw=1.3),
             bbox=dict(facecolor='#D5F5E3', edgecolor='#27AE60',
                       boxstyle='round,pad=0.3', alpha=0.85))

ax2.text(0, -1.1, '2D高斯片（表面表示）\n贴合表面，几何更精确',
         ha='center', va='bottom', fontsize=10, color='#2C3E50',
         bbox=dict(facecolor='#EAFAF1', edgecolor='#82E0AA',
                   boxstyle='round,pad=0.4', alpha=0.9))

# 中间分隔 + 标题
fig.text(0.5, 0.96, '图9.2  2DGS 与 3DGS 几何表示对比',
         ha='center', va='top', fontsize=13, fontweight='bold', color='#2C3E50')

# 添加 VS 标签
fig.text(0.5, 0.52, 'VS', ha='center', va='center',
         fontsize=18, fontweight='bold', color='#AAB7B8')

plt.subplots_adjust(wspace=0.08, top=0.90)
plt.savefig('docs/pic/ch9_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch9_fig2.png saved")


# ── 图9.3 各方法 FPS vs PSNR 散点图 ────────────────────────────────
methods = [
    ('NeRF\n(基准)',       0.008, 27.0,  '#7F8C8D', 90),
    ('Mip-NeRF 360',       0.02,  29.5,  '#7F8C8D', 90),
    ('Instant-NGP',        8.0,   28.5,  '#3498DB', 90),
    ('3DGS',               137,   28.2,  '#E74C3C', 150),
    ('Scaffold-GS',        95,    29.1,  '#E74C3C', 120),
    ('Mini-Splatting',     140,   25.8,  '#F39C12', 90),
    ('Compact-3DGS',       110,   27.9,  '#27AE60', 90),
    ('2DGS',               60,    28.0,  '#9B59B6', 100),
    ('LightGaussian',      130,   27.5,  '#1ABC9C', 90),
]

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('#F9F9F9')

# 理想区域背景
ax.fill_betweenx([28.8, 30.2], 70, 300, alpha=0.10, color='#27AE60')
ax.text(185, 30.05, '理想区域\n（高质量 + 快速）', fontsize=9.5, color='#1E8449',
        ha='center', va='center',
        bbox=dict(facecolor='#EAFAF1', edgecolor='#82E0AA',
                  boxstyle='round,pad=0.3', alpha=0.8))

# "更好"方向箭头
ax.annotate('', xy=(250, 30.3), xytext=(15, 26.5),
            arrowprops=dict(arrowstyle='->', color='#E74C3C',
                            lw=2.0, connectionstyle='arc3,rad=-0.15'))
ax.text(130, 26.2, '更好 →', fontsize=10, color='#E74C3C',
        rotation=-18, fontweight='bold')

for name, fps, psnr, color, size in methods:
    ax.scatter(fps, psnr, s=size, color=color, zorder=5, alpha=0.88,
               edgecolors='white', linewidths=0.8)
    # 偏移策略
    if fps < 0.1:
        dx, dy, ha = 0.012, 0.12, 'left'
    elif fps < 1:
        dx, dy, ha = 0.5, 0.12, 'left'
    elif fps < 30:
        dx, dy, ha = 1.5, 0.12, 'left'
    elif fps > 110:
        dx, dy, ha = -2, -0.22, 'right'
    else:
        dx, dy, ha = 2, 0.12, 'left'
    ax.annotate(name, (fps, psnr), xytext=(fps * (1 + dx * 0.05), psnr + dy),
                fontsize=8.5, ha=ha, color=color,
                bbox=dict(facecolor='white', edgecolor=color,
                          boxstyle='round,pad=0.15', alpha=0.7, lw=0.8))

ax.axvline(30, color='#AAB7B8', ls='--', lw=1.5, alpha=0.7)
ax.text(32, 25.2, '实时 30FPS', fontsize=9, color='#7F8C8D')

ax.set_xscale('log')
ax.set_xlabel('渲染速度（FPS，对数坐标）', fontsize=12)
ax.set_ylabel('PSNR (dB)  ↑ 越高越好', fontsize=12)
ax.set_title('图9.3  各主要方法渲染速度 vs 图像质量对比', fontsize=13,
             fontweight='bold', pad=10)
ax.set_xlim(0.005, 400)
ax.set_ylim(24.5, 30.8)
ax.grid(alpha=0.3, which='both')

# 图例：颜色分类
legend_handles = [
    mpatches.Patch(facecolor='#7F8C8D', label='NeRF 系列'),
    mpatches.Patch(facecolor='#3498DB', label='混合表示'),
    mpatches.Patch(facecolor='#E74C3C', label='3DGS 原版及改进'),
    mpatches.Patch(facecolor='#27AE60', label='压缩优化'),
    mpatches.Patch(facecolor='#9B59B6', label='几何改进'),
    mpatches.Patch(facecolor='#F39C12', label='极简化'),
    mpatches.Patch(facecolor='#1ABC9C', label='轻量化'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='lower left',
          framealpha=0.9, edgecolor='#CCC')

plt.tight_layout()
plt.savefig('docs/pic/ch9_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch9_fig3.png saved")


# ── 图9.4 论文发表数量时间线（2020-2026）────────────────────────────
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
nerf_papers = [5,   18,  45,  30,  20,  12,   8]
gs_papers   = [0,    0,   0,   8,  85, 130,  85]

x = np.arange(len(years))
w = 0.38

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_facecolor('#FAFAFA')

bars1 = ax.bar(x - w / 2, nerf_papers, w, label='NeRF 系列',
               color='#4C72B0', alpha=0.87, edgecolor='white', lw=0.8)
bars2 = ax.bar(x + w / 2, gs_papers,   w, label='3DGS 系列',
               color='#E74C3C', alpha=0.87, edgecolor='white', lw=0.8)

# 数值标签
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                str(int(h)), ha='center', va='bottom', fontsize=8.5, color='#2C3E50')
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                str(int(h)), ha='center', va='bottom', fontsize=8.5, color='#2C3E50')

# 里程碑标注
milestones = [
    (0,  nerf_papers[0],  'NeRF\n(ECCV 2020)',      -w/2, '#4C72B0'),
    (3,  gs_papers[3],    '3DGS\n(SIGGRAPH 2023)',  +w/2, '#E74C3C'),
    (4,  gs_papers[4],    '2DGS / Scaffold-GS\n爆发期', +w/2, '#C0392B'),
]
for xi, yi, label, offset, color in milestones:
    ax.annotate(label,
                xy=(xi + offset, yi),
                xytext=(xi + offset, yi + 18),
                fontsize=8.5, ha='center', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3),
                bbox=dict(facecolor='white', edgecolor=color,
                          boxstyle='round,pad=0.3', alpha=0.85))

# 增长趋势注释
ax.text(5.55, 115, '2025 年\n仍快速增长', fontsize=9, color='#E74C3C',
        ha='center', style='italic',
        bbox=dict(facecolor='#FEF9E7', edgecolor='#F9CA24',
                  boxstyle='round,pad=0.3', alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years], fontsize=11)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('相关论文数量（估计）', fontsize=12)
ax.set_title('图9.4  NeRF 与 3DGS 系列论文发表数量时间线（2020—2026）',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 165)

plt.tight_layout()
plt.savefig('docs/pic/ch9_fig4.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch9_fig4.png saved")

print("ch9 all 4 figures done")
