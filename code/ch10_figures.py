import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('docs/pic', exist_ok=True)

# ── 图10.1 实验管理工作流 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.axis('off')
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)

# 主流程步骤: (cx, cy, label, face_color, tool_label)
steps = [
    (1.4,  4.2, '想法/\n假设',             '#B0BEC5', '想法记录\nNotion/README'),
    (3.4,  4.2, '修改代码\ngit commit',    '#90CAF9', 'Git\n版本控制'),
    (5.6,  4.2, '运行实验\npython train.py','#A5D6A7', 'YAML\n配置文件'),
    (7.9,  4.2, '记录结果\nWandb/TBoard',  '#FFE082', 'Wandb\nTensorBoard'),
    (10.2, 4.2, '分析对比\nbaseline vs exp','#CE93D8', '对比表格\n统计检验'),
]

BOX_W = 2.0
BOX_H = 1.0

# 绘制主流程框
for cx, cy, label, color, _ in steps:
    box = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                         boxstyle='round,pad=0.12',
                         facecolor=color, edgecolor='#546E7A', lw=1.8)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=10.5, fontweight='bold')

# 主流程箭头（灰色→下一步）
for i in range(len(steps) - 1):
    x1 = steps[i][0] + BOX_W/2
    x2 = steps[i+1][0] - BOX_W/2
    ax.annotate('', xy=(x2 + 0.04, 4.2), xytext=(x1 - 0.04, 4.2),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=2.2,
                                mutation_scale=18))

# 工具标注（步骤下方）
for cx, cy, _, _, tool_label in steps:
    ax.text(cx, cy - 1.1, tool_label, ha='center', va='center', fontsize=9,
            color='#37474F',
            bbox=dict(facecolor='#FAFAFA', edgecolor='#B0BEC5',
                      boxstyle='round,pad=0.18', lw=1.2))
    ax.annotate('', xy=(cx, cy - 0.5 - 0.02), xytext=(cx, cy - 0.7),
                arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2,
                                mutation_scale=12))

# 绿色"改进"成功分支（右端向右延伸）
ax.annotate('', xy=(12.5, 4.2), xytext=(11.2, 4.2),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.2,
                            mutation_scale=18))
succ_box = FancyBboxPatch((12.5, 3.7), 2.2, 1.0,
                           boxstyle='round,pad=0.12',
                           facecolor='#C8E6C9', edgecolor='#2E7D32', lw=2.0)
ax.add_patch(succ_box)
ax.text(13.6, 4.2, '发现规律\n总结成果', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#1B5E20')

# 红色"失败/改进"回环
# 从 "分析对比" 底部出发 → 弧形回到 "修改代码" 底部
ax.annotate('', xy=(3.4, 2.35), xytext=(10.2, 2.35),
            arrowprops=dict(arrowstyle='<-', color='#C62828', lw=2.2,
                            mutation_scale=18))
ax.text(6.8, 1.85, '发现问题 → 继续迭代（红色回路）',
        ha='center', va='center', fontsize=10, color='#C62828',
        bbox=dict(facecolor='#FFEBEE', edgecolor='#EF9A9A',
                  boxstyle='round,pad=0.2', lw=1.2))
# 竖线：分析对比底 → 回路
ax.plot([10.2, 10.2], [3.7, 2.35], color='#C62828', lw=2.0)
# 竖线：修改代码底 → 回路
ax.plot([3.4, 3.4], [3.7, 2.35], color='#C62828', lw=2.0)

# 颜色图例
legend_items = [
    mpatches.Patch(facecolor='#B0BEC5', edgecolor='#546E7A', label='常规步骤（灰色）'),
    mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='成功/发现（绿色）'),
    mpatches.Patch(facecolor='#FFEBEE', edgecolor='#EF9A9A', label='失败/迭代（红色）'),
]
ax.legend(handles=legend_items, loc='upper right',
          fontsize=9.5, framealpha=0.9, edgecolor='#B0BEC5')

ax.set_title('图10.1  可复现实验管理工作流', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('docs/pic/ch10_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch10_fig1.png saved")


# ── 图10.2 GPU Profiling 甘特图 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

# (内核名称, 开始ms, 结束ms, 颜色, 类型标签)
kernels = [
    ('前向渲染（光栅化预处理）',   0.00, 4.20,  '#1565C0', '计算核'),
    ('高斯投影与排序准备',         4.20, 6.80,  '#1976D2', '计算核'),
    ('Radix Sort（ADC排序）',      6.80, 8.50,  '#F57F17', '内存操作'),
    ('Alpha合成（Tile分块）',      8.50, 13.50, '#0277BD', '计算核'),
    ('显存复制 Device→Host',      13.50, 15.00, '#78909C', '内存操作'),
    ('损失计算（L1 + D-SSIM）',   15.00, 17.20, '#2E7D32', '计算核'),
    ('反向传播（光栅化梯度）',    17.20, 30.00, '#B71C1C', '计算核'),  # ← 瓶颈
    ('高斯参数梯度累积',          30.00, 33.50, '#C62828', '计算核'),
    ('Adam优化器参数更新',        33.50, 36.80, '#6A1B9A', '计算核'),
    ('剪枝/克隆/密度控制',        36.80, 39.50, '#E65100', '计算核'),
    ('显存复制 Host→Device',      39.50, 41.00, '#78909C', '内存操作'),
]

type_colors = {'计算核': None, '内存操作': '#78909C'}

for i, (name, t_start, t_end, color, ktype) in enumerate(kernels):
    duration = t_end - t_start
    bar = ax.barh(i, duration, left=t_start, color=color,
                  height=0.62, alpha=0.88, edgecolor='white', linewidth=0.8)
    # 持续时间标注
    mid = (t_start + t_end) / 2
    ax.text(mid, i, f'{duration:.1f}ms', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

# Y轴标签
ax.set_yticks(range(len(kernels)))
ax.set_yticklabels([k[0] for k in kernels], fontsize=9.5)
ax.set_xlabel('时间（ms）', fontsize=11)
ax.set_xlim(-1, 44)
ax.set_ylim(-0.7, len(kernels) - 0.3)
ax.invert_yaxis()

# 瓶颈标注
bottleneck_idx = 6  # 反向传播
ax.annotate('瓶颈：反向传播占\n总时间约31%',
            xy=(23.6, bottleneck_idx),
            xytext=(36, bottleneck_idx + 2.2),
            fontsize=9.5, color='#B71C1C',
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.6),
            bbox=dict(facecolor='#FFEBEE', edgecolor='#EF9A9A',
                      boxstyle='round,pad=0.3', lw=1.2))

# 总耗时线
total = kernels[-1][2]
ax.axvline(total, color='#424242', ls='--', lw=1.4, alpha=0.6)
ax.text(total + 0.3, -0.5, f'总计\n{total:.0f}ms', fontsize=8.5,
        color='#424242', va='top')

# 图例
legend_handles = [
    mpatches.Patch(facecolor='#1565C0', alpha=0.88, label='计算核（蓝/绿/紫系）'),
    mpatches.Patch(facecolor='#B71C1C', alpha=0.88, label='反向传播（瓶颈，红色）'),
    mpatches.Patch(facecolor='#78909C', alpha=0.88, label='内存操作（灰色）'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#B0BEC5')

ax.set_title('图10.2  3DGS训练单步 GPU Profiling 时间分布（甘特图示意）',
             fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig('docs/pic/ch10_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch10_fig2.png saved")


# ── 图10.3 3DGS社区生态图（同心圆布局）────────────────────────────
fig, ax = plt.subplots(figsize=(13, 12))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 7.0)

def draw_node(ax, cx, cy, label, desc, facecolor, edgecolor, fontsize=9, width=2.6, height=1.05):
    box = FancyBboxPatch((cx - width/2, cy - height/2), width, height,
                         boxstyle='round,pad=0.15',
                         facecolor=facecolor, edgecolor=edgecolor, lw=2.0, zorder=3)
    ax.add_patch(box)
    full_text = f'{label}\n{desc}' if desc else label
    ax.text(cx, cy, full_text, ha='center', va='center',
            fontsize=fontsize, zorder=4,
            linespacing=1.35)

def draw_arrow(ax, x1, y1, x2, y2, color='#90A4AE', lw=1.5, style='solid', zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14,
                                linestyle=style,
                                connectionstyle='arc3,rad=0.05'),
                zorder=zorder)

# ── 中心节点
draw_node(ax, 0, 0,
          'gaussian-splatting\n(INRIA官方)',
          '2023年原版实现\n实时渲染基准',
          facecolor='#FF6B35', edgecolor='#BF360C',
          fontsize=10, width=3.0, height=1.4)

# ── 第一圈：核心工程库（半径 ~3.0）
r1 = 3.2
tier1_nodes = [
    # (角度°, label, desc, face, edge)
    (90,  'gsplat\n(Nerfstudio团队)',   '高性能CUDA后端\n支持多GPU训练',  '#1565C0', '#0D47A1'),
    (30,  'nerfstudio',                 '模块化训练框架\nSplatfacto内置', '#2E7D32', '#1B5E20'),
    (330, '3DGUT (NVIDIA)',             '非标准相机支持\n2025年发布',     '#6A1B9A', '#4A148C'),
    (210, 'Mip-Splatting',             '抗混叠改进版\n多尺度高斯',       '#00838F', '#006064'),
    (150, 'Scaffold-GS',               '锚点结构引导\n场景表示更紧凑',   '#558B2F', '#33691E'),
    (270, '4D Gaussian',               '时序动态场景\n视频重建',          '#AD1457', '#880E4F'),
]

tier1_positions = []
for angle_deg, label, desc, face, edge in tier1_nodes:
    angle_rad = np.radians(angle_deg)
    cx = r1 * np.cos(angle_rad)
    cy = r1 * np.sin(angle_rad)
    tier1_positions.append((cx, cy))
    draw_node(ax, cx, cy, label, desc, facecolor=face, edgecolor=edge,
              fontsize=8.5, width=2.7, height=1.0)
    # 从中心指向第一圈的箭头
    # 箭头起点：中心节点边缘
    scale_start = 0.75
    scale_end = 0.82
    draw_arrow(ax,
               cx * (1 - scale_end), cy * (1 - scale_end),
               cx * (1 - scale_start * 0.24), cy * (1 - scale_start * 0.24),
               color=edge, lw=1.8)

# 重新计算正确的箭头端点
for (angle_deg, label, desc, face, edge), (cx, cy) in zip(tier1_nodes, tier1_positions):
    # 修正：从中心节点外沿到第一圈节点内沿
    dist = np.sqrt(cx**2 + cy**2)
    ux, uy = cx / dist, cy / dist
    # 中心节点半径约0.7（宽1.5高0.7）
    src_x = ux * 0.76
    src_y = uy * 0.52
    # 第一圈节点内沿
    dst_x = cx - ux * 1.35
    dst_y = cy - uy * 0.52
    ax.annotate('', xy=(dst_x, dst_y), xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color=edge, lw=1.8,
                                mutation_scale=14,
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)

# ── 第二圈：应用工具（半径 ~5.3）
r2 = 5.4
tier2_nodes = [
    (72,  'SIBR Viewer',   '官方实时\n可视化工具',  '#E53935', '#B71C1C'),
    (144, 'SuperSplat',    '网页端\n编辑器',         '#F57C00', '#E65100'),
    (216, 'Luma AI',       '商业SaaS\n端到端服务',   '#00897B', '#004D40'),
    (288, 'Polycam',       '移动端\n3D扫描App',      '#039BE5', '#01579B'),
    (0,   'three.js插件',  'WebGL实时\n网页渲染',    '#7B1FA2', '#4A148C'),
]

for angle_deg, label, desc, face, edge in tier2_nodes:
    angle_rad = np.radians(angle_deg)
    cx = r2 * np.cos(angle_rad)
    cy = r2 * np.sin(angle_rad)
    draw_node(ax, cx, cy, label, desc, facecolor=face, edgecolor=edge,
              fontsize=8, width=2.3, height=0.95)
    # 虚线从第一圈最近节点到第二圈（或从中心连出）
    dist = np.sqrt(cx**2 + cy**2)
    ux, uy = cx / dist, cy / dist
    src_x = ux * 1.6
    src_y = uy * 1.6
    dst_x = cx - ux * 1.15
    dst_y = cy - uy * 0.5
    ax.annotate('', xy=(dst_x, dst_y), xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color=edge, lw=1.4,
                                mutation_scale=12, linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'),
                zorder=1)

# 同心圆辅助线（虚线圆）
for r, label, ls in [(r1, '核心工程库', '--'), (r2, '应用工具', ':')]:
    circle = plt.Circle((0, 0), r, fill=False, linestyle=ls,
                         edgecolor='#CFD8DC', linewidth=1.2, zorder=0)
    ax.add_patch(circle)
    ax.text(r * np.cos(np.radians(45)) + 0.1,
            r * np.sin(np.radians(45)) + 0.3,
            label, fontsize=9, color='#90A4AE',
            ha='left', va='bottom')

# 图例
legend_items = [
    mpatches.Patch(facecolor='#FF6B35', edgecolor='#BF360C', label='核心原版（INRIA）'),
    mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='第一圈：核心工程库/改进算法'),
    mpatches.Patch(facecolor='#E53935', edgecolor='#B71C1C', label='第二圈：应用工具/商业产品'),
]
ax.legend(handles=legend_items, loc='lower center',
          fontsize=9.5, framealpha=0.95, edgecolor='#B0BEC5',
          bbox_to_anchor=(0.5, 0.01), ncol=3)

ax.set_title('图10.3  3DGS 开源社区生态图（截至2026年中）',
             fontsize=14, fontweight='bold')
ax.text(0, -6.1,
        '实线箭头：基于/派生关系　虚线箭头：调用/依赖关系\n内圈：核心算法库　外圈：应用工具',
        ha='center', va='center', fontsize=9.5, color='#546E7A')

plt.savefig('docs/pic/ch10_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch10_fig3.png saved")

print("ch10 all done")
