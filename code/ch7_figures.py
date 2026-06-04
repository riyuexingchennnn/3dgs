import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图7.1 代码仓库文件树 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)

# (x_indent, y, text, color, is_folder, annotation)
tree_items = [
    (0.3, 14.2, 'gaussian-splatting/',                    '#2C3E50', True,  '项目根目录'),
    (0.9, 13.4, '├── train.py',                           '#E74C3C', False, '训练入口，主循环控制'),
    (0.9, 12.6, '├── render.py',                          '#E74C3C', False, '渲染测试集，保存图像'),
    (0.9, 11.8, '├── metrics.py',                         '#E74C3C', False, '计算PSNR / SSIM / LPIPS'),
    (0.9, 11.0, '├── scene/',                             '#2980B9', True,  '场景管理模块'),
    (1.7, 10.2, '│   ├── gaussian_model.py',              '#27AE60', False, 'GaussianModel类，存储高斯参数'),
    (1.7,  9.4, '│   └── cameras.py',                     '#27AE60', False, '相机模型与投影变换'),
    (0.9,  8.6, '├── gaussian_renderer/',                 '#2980B9', True,  '可微分渲染器封装'),
    (1.7,  7.8, '│   └── __init__.py',                    '#27AE60', False, 'render()函数，调用CUDA光栅化'),
    (0.9,  7.0, '├── utils/',                             '#2980B9', True,  '通用工具函数'),
    (1.7,  6.2, '│   ├── loss_utils.py',                  '#27AE60', False, 'L1 + SSIM 损失'),
    (1.7,  5.4, '│   └── sh_utils.py',                    '#27AE60', False, '球谐函数辅助'),
    (0.9,  4.6, '└── submodules/',                        '#2980B9', True,  'C++/CUDA 扩展子模块'),
    (1.7,  3.8, '    └── diff-gaussian-rasterization/',   '#8E44AD', True,  'CUDA 可微分光栅化核心'),
    (2.5,  3.0, '        ├── forward.cu',                 '#C0392B', False, '前向渲染CUDA核心'),
    (2.5,  2.2, '        └── backward.cu',                '#C0392B', False, '反向传播梯度计算'),
]

for x, y, text, color, is_folder, note in tree_items:
    weight = 'bold' if is_folder else 'normal'
    ax.text(x, y, text, fontsize=10, color=color, fontweight=weight,
            fontfamily='monospace', va='center')
    if note:
        ax.text(x + len(text) * 0.135 + 0.3, y, f'  # {note}',
                fontsize=8.5, color='#95A5A6', va='center', style='italic')

# 图例
folder_patch = mpatches.Patch(color='#2980B9', label='目录')
file_patch   = mpatches.Patch(color='#E74C3C', label='Python 文件')
cuda_patch   = mpatches.Patch(color='#C0392B', label='CUDA 文件')
py_patch     = mpatches.Patch(color='#27AE60', label='模块文件')
ax.legend(handles=[folder_patch, file_patch, py_patch, cuda_patch],
          loc='lower right', fontsize=9, framealpha=0.8)

ax.set_title('图7.1  gaussian-splatting 代码仓库目录结构', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('docs/pic/ch7_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch7_fig1.png saved")

# ── 图7.2 train.py主循环流程图 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 13))
ax.axis('off')
ax.set_xlim(0, 7)
ax.set_ylim(0, 13)

# (cx, cy, label, facecolor, box_w, box_h, dashed)
steps = [
    (3.5, 12.2, '初始化场景\nScene + GaussianModel',            '#AEC6CF', 3.0, 0.75, False),
    (3.5, 11.0, '设置 Adam 优化器',                              '#AEC6CF', 2.8, 0.6,  False),
    (3.5,  9.7, 'for i in range(30000):',                       '#FFD1DC', 3.4, 0.65, True),  # loop header
    (3.5,  8.7, '随机选取训练视角',                              '#FFDAC1', 2.8, 0.6,  False),
    (3.5,  7.65,'前向渲染\nrender(camera, gaussians)',           '#B5EAD7', 3.2, 0.75, False),
    (3.5,  6.55,'计算 Loss\n(1−λ)·L1 + λ·D-SSIM',             '#B5EAD7', 3.2, 0.75, False),
    (3.5,  5.45,'loss.backward()\n反向传播',                    '#B5EAD7', 3.0, 0.75, False),
    (3.5,  4.35,'optimizer.step()\n更新参数',                   '#B5EAD7', 3.0, 0.75, False),
    (3.5,  3.2, '[每 100 步]  ADC\n克隆 / 分裂 / 剪枝高斯',     '#FFDAC1', 3.2, 0.75, True),   # special dashed
    (3.5,  2.1, '[每 1000 步]  保存 checkpoint',                '#C7CEEA', 3.4, 0.65, True),   # special dashed
    (3.5,  1.0, '输出最终模型  point_cloud.ply',                '#AEC6CF', 3.2, 0.65, False),
]

# 循环虚线大框：从 for 行顶部到 checkpoint 底部
loop_top    = steps[2][1] + steps[2][5]/2 + 0.15
loop_bottom = steps[9][1] - steps[9][5]/2 - 0.15
loop_left   = 3.5 - 3.8/2 - 0.15
loop_right  = 3.5 + 3.8/2 + 0.15
loop_rect = plt.Polygon(
    [[loop_left,  loop_bottom],
     [loop_right, loop_bottom],
     [loop_right, loop_top],
     [loop_left,  loop_top]],
    closed=True, fill=False,
    edgecolor='#3498DB', linewidth=1.8, linestyle='--', zorder=0)
ax.add_patch(loop_rect)
ax.text(loop_left - 0.05, (loop_top + loop_bottom) / 2,
        '训\n练\n循\n环', fontsize=9, color='#3498DB',
        ha='right', va='center')

for i, step in enumerate(steps):
    cx, cy, label, fc, bw, bh, dashed = step
    ec  = '#E74C3C' if dashed else '#888888'
    lw  = 2.0       if dashed else 1.5
    ls  = '--'      if dashed else '-'
    box = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                         boxstyle='round,pad=0.08',
                         facecolor=fc, edgecolor=ec, lw=lw,
                         linestyle=ls, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=9.5, zorder=3)
    if i < len(steps) - 1:
        next_top = steps[i+1][1] + steps[i+1][5]/2
        cur_bot  = cy - bh/2
        ax.annotate('', xy=(cx, next_top + 0.02), xytext=(cx, cur_bot - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5),
                    zorder=4)

ax.set_title('图7.2  train.py 主循环流程图', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig('docs/pic/ch7_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch7_fig2.png saved")

# ── 图7.3 NeRF vs 3DGS 性能雷达图 ─────────────────────────────
labels = ['训练速度', '渲染速度', '图像质量', '内存效率', '场景编辑性']
N      = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 虚构但符合实际认知的评分 (0-10)
nerf_vals = [2, 2, 8, 7, 3];  nerf_vals += nerf_vals[:1]
gs_vals   = [8, 10, 8, 4, 7]; gs_vals   += gs_vals[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='#666666')
ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.8)

# NeRF — 蓝色
ax.plot(angles, nerf_vals, color='#2980B9', linewidth=2.5, linestyle='-', label='NeRF')
ax.fill(angles, nerf_vals, color='#2980B9', alpha=0.18)

# 3DGS — 橙色
ax.plot(angles, gs_vals, color='#E67E22', linewidth=2.5, linestyle='-', label='3DGS')
ax.fill(angles, gs_vals, color='#E67E22', alpha=0.18)

# 数值标注
for angle, nv, gv in zip(angles[:-1], nerf_vals[:-1], gs_vals[:-1]):
    ax.text(angle, nv + 0.6, str(nv), ha='center', va='center',
            fontsize=9, color='#2980B9', fontweight='bold')
    ax.text(angle, gv + 0.6, str(gv), ha='center', va='center',
            fontsize=9, color='#E67E22', fontweight='bold')

ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=12,
          framealpha=0.85)
ax.set_title('图7.3  NeRF vs 3DGS 性能对比雷达图\n（各维度 0–10 分）',
             fontsize=12, pad=22)
plt.tight_layout()
plt.savefig('docs/pic/ch7_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch7_fig3.png saved")
print("ch7 all done")
