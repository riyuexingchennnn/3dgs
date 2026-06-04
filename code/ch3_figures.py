import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图3.1 一维高斯函数对比 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x = np.linspace(-3.5, 5.5, 600)

def gauss1d(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

configs = [
    (0.0, 0.5, '#E74C3C', '均值=0，标准差=0.5'),
    (0.0, 1.0, '#3498DB', '均值=0，标准差=1.0'),
    (2.0, 0.8, '#2ECC71', '均值=2，标准差=0.8'),
]

for mu, sigma, color, label in configs:
    y = gauss1d(x, mu, sigma)
    ax.plot(x, y, color=color, lw=2.5, label=label)
    # 均值位置竖线
    y_peak = gauss1d(mu, mu, sigma)
    ax.axvline(x=mu, color=color, lw=1.3, linestyle='--', alpha=0.7)
    ax.text(mu + 0.07, y_peak * 0.52, f'μ={mu}', fontsize=9, color=color,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('概率密度 f(x)', fontsize=12)
ax.set_title('图3.1 一维高斯函数对比', fontsize=14)
ax.legend(fontsize=10.5, loc='upper right')
ax.set_xlim(-3.5, 5.5)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/pic/ch3_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch3_fig1.png saved")

# ── 图3.2 二维高斯等高线图（3子图）────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

x_lin = np.linspace(-4, 4, 250)
y_lin = np.linspace(-4, 4, 250)
X, Y = np.meshgrid(x_lin, y_lin)

def gauss2d_cov(X, Y, Sigma):
    """2D Gaussian given 2x2 covariance matrix Sigma, zero mean."""
    inv_S = np.linalg.inv(Sigma)
    det_S = np.linalg.det(Sigma)
    pos = np.stack([X, Y], axis=-1)
    exponent = -0.5 * np.einsum('...i,ij,...j->...', pos, inv_S, pos)
    norm = 2 * np.pi * np.sqrt(det_S)
    return np.exp(exponent) / norm

cmap_list = ['Blues', 'Oranges', 'Greens']
sigma_configs = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]),  '圆形高斯\n(σx=σy=1，无相关)'),
    (np.array([[4.0, 0.0], [0.0, 0.64]]), '椭圆高斯\n(σx=2，σy=0.8，无相关)'),
    (np.array([[2.25, 1.2], [1.2, 1.0]]), '旋转椭圆\n(有相关性，ρ≈0.8)'),
]

for ax, (Sigma, title), cmap in zip(axes, sigma_configs, cmap_list):
    Z = gauss2d_cov(X, Y, Sigma)
    cf = ax.contourf(X, Y, Z, levels=12, cmap=cmap)
    ax.contour(X, Y, Z, levels=12, colors='white', linewidths=0.5, alpha=0.5)
    ax.plot(0, 0, 'r+', ms=12, mew=2.5, label='均值')
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)

fig.suptitle('图3.2 二维高斯等高线图（等概率密度轮廓）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('docs/pic/ch3_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch3_fig2.png saved")

# ── 图3.3 协方差矩阵分解示意 ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

theta = np.linspace(0, 2 * np.pi, 400)

# 步骤1：单位圆
pts_unit = np.vstack([np.cos(theta), np.sin(theta)])

# 步骤2：缩放 S = diag(2.2, 0.8)
sx, sy = 2.2, 0.8
S = np.diag([sx, sy])
pts_scaled = S @ pts_unit

# 步骤3：旋转 R（40度）
angle_deg = 40
angle_rad = np.deg2rad(angle_deg)
R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
              [np.sin(angle_rad),  np.cos(angle_rad)]])
pts_rotated = R @ pts_scaled

step_data = [
    (pts_unit,   '#4C72B0', '步骤1：单位圆\n（初始形状）',
     f'I（单位矩阵）'),
    (pts_scaled, '#DD8452', f'步骤2：缩放 S\ndiag({sx}, {sy})',
     f'S = diag({sx}, {sy})'),
    (pts_rotated, '#55A868', f'步骤3：旋转 R（{angle_deg}°）\n旋转椭圆',
     'Σ = R·S·Sᵀ·Rᵀ'),
]

for ax, (pts, color, title, annotation) in zip(axes, step_data):
    ax.fill(pts[0], pts[1], alpha=0.28, color=color)
    ax.plot(pts[0], pts[1], color=color, lw=2.5)
    ax.plot(0, 0, 'k+', ms=10, mew=2)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.7, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.7, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.10, annotation, transform=ax.transAxes,
            ha='center', fontsize=10, color=color,
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'))
    ax.grid(True, alpha=0.15)

# 步骤2：标注缩放轴
axes[1].annotate('', xy=(sx, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#DD8452', lw=2))
axes[1].text(sx / 2, 0.22, f'sx={sx}', fontsize=9, color='#DD8452', ha='center')
axes[1].annotate('', xy=(0, sy), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))
axes[1].text(0.22, sy / 2, f'sy={sy}', fontsize=9, color='#C0392B')

# 步骤3：标注旋转弧
arc_t = np.linspace(0, angle_rad, 60)
arc_r = 0.9
axes[2].plot(arc_r * np.cos(arc_t), arc_r * np.sin(arc_t), 'purple', lw=2)
axes[2].annotate('', xy=(arc_r * np.cos(angle_rad), arc_r * np.sin(angle_rad)),
                 xytext=(arc_r * np.cos(angle_rad * 0.95),
                         arc_r * np.sin(angle_rad * 0.95)),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
axes[2].text(0.75, 0.62, f'{angle_deg}°', fontsize=9, color='purple')

# 子图间箭头（图形坐标）
for xpos, label in [(0.355, 'S（缩放）'), (0.685, 'R（旋转）')]:
    fig.text(xpos, 0.50, '⟶', fontsize=26, ha='center', va='center', color='#555')
    fig.text(xpos, 0.34, label, fontsize=10, ha='center', va='center', color='#555')

fig.suptitle('图3.3 协方差矩阵分解：Σ = R·S·Sᵀ·Rᵀ', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('docs/pic/ch3_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch3_fig3.png saved")

# ── 图3.4 高斯混合示意（1D）──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.linspace(-5, 10, 600)

components = [
    (-1.5, 0.8,  0.35, '#E74C3C', '分量1（μ=-1.5，σ=0.8）'),
    ( 3.0, 1.2,  0.40, '#3498DB', '分量2（μ=3.0，σ=1.2）'),
    ( 7.0, 0.75, 0.25, '#2ECC71', '分量3（μ=7.0，σ=0.75）'),
]

mixture = np.zeros_like(x)
for mu, sigma, weight, color, label in components:
    y = weight * gauss1d(x, mu, sigma)
    ax.fill_between(x, y, alpha=0.22, color=color)
    ax.plot(x, y, color=color, lw=2.2, label=label)
    mixture += y

ax.plot(x, mixture, 'k-', lw=3.2, label='混合分布（加权叠加）')

# 竖线标注各分量均值
for mu, sigma, weight, color, _ in components:
    ax.axvline(x=mu, color=color, lw=1.0, linestyle=':', alpha=0.75)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('图3.4 一维高斯混合分布示意', fontsize=14)
ax.legend(fontsize=10.5, loc='upper right')
ax.set_xlim(-5, 10)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/pic/ch3_fig4.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch3_fig4.png saved")

print("ch3 all done")
