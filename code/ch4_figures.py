import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('docs/pic', exist_ok=True)

# ── 图4.1 漫反射 vs 镜面反射示意 ────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

def draw_surface(ax):
    """画水平表面和法线"""
    ax.plot([-2.2, 2.2], [0, 0], color='#444', lw=3, zorder=1)
    # 表面阴影纹理
    for x in np.arange(-2.0, 2.2, 0.3):
        ax.plot([x, x + 0.2], [0, -0.18], color='#999', lw=1, zorder=1)
    ax.text(0, -0.38, '表面', ha='center', fontsize=10, color='#555')
    # 法线（虚线）
    ax.annotate('', xy=(0, 1.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5,
                                linestyle='dashed', connectionstyle='arc3'))
    ax.text(0.1, 1.85, '法线 N', fontsize=9, color='#777')

def angle_arc(ax, cx, cy, r, a1_deg, a2_deg, color, label):
    """绘制角度弧 + 文字"""
    ts = np.linspace(np.radians(a1_deg), np.radians(a2_deg), 60)
    ax.plot(cx + r * np.cos(ts), cy + r * np.sin(ts), color=color, lw=1.5)
    mid = np.radians((a1_deg + a2_deg) / 2)
    ax.text(cx + (r + 0.22) * np.cos(mid), cy + (r + 0.22) * np.sin(mid),
            label, fontsize=9, color=color, ha='center', va='center')

# ---------- 左子图：漫反射 ----------
ax = axes[0]
draw_surface(ax)

# 入射光（左上 -> 原点，角度135°）
ax.annotate('', xy=(0, 0), xytext=(-1.4, 1.4),
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.5))
ax.text(-1.55, 1.55, '入射光', fontsize=10, color='#E67E22')
angle_arc(ax, 0, 0, 0.52, 90, 135, '#E67E22', '入射角 θᵢ')

# 漫反射：13条均匀分布射线
for theta in np.linspace(10, 170, 13):
    r_val = np.radians(theta)
    # 朗伯余弦调制亮度
    length = 0.95 + 0.35 * np.sin(r_val)
    ex, ey = np.cos(r_val) * length, np.sin(r_val) * length
    alpha_val = float(0.35 + 0.6 * np.sin(r_val))
    ax.annotate('', xy=(ex, ey), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#3498DB',
                                lw=1.6, alpha=alpha_val))

# 半圆弧（示意均匀半球）
arc_t = np.linspace(0, np.pi, 120)
ax.plot(0.75 * np.cos(arc_t), 0.75 * np.sin(arc_t),
        '--', color='#3498DB', lw=1.3, alpha=0.55)

ax.text(0, 2.35, '均匀向各方向散射\n（朗伯余弦定律）',
        ha='center', fontsize=9.5, color='#3498DB', style='italic')
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.6, 2.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('漫反射（Diffuse Reflection）', fontsize=12, pad=8)

# ---------- 右子图：镜面反射 ----------
ax = axes[1]
draw_surface(ax)

# 入射光（左上 -> 原点，角度135°）
ax.annotate('', xy=(0, 0), xytext=(-1.4, 1.4),
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.5))
ax.text(-1.55, 1.55, '入射光', fontsize=10, color='#E67E22')
angle_arc(ax, 0, 0, 0.52, 90, 135, '#E67E22', '入射角 θᵢ')

# 主镜面反射光（右上，角度45°，与入射角对称）
ax.annotate('', xy=(1.4, 1.4), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3.0))
ax.text(1.1, 1.55, '镜面反射', fontsize=10, color='#E74C3C')
angle_arc(ax, 0, 0, 0.52, 45, 90, '#E74C3C', '反射角 θᵣ')

# 扩散光（高光波瓣，越偏越弱）
for dtheta, alp in [(-22, 0.22), (-12, 0.52), (12, 0.52), (22, 0.22)]:
    r_val = np.radians(45 + dtheta)
    ex, ey = np.cos(r_val) * 1.05, np.sin(r_val) * 1.05
    ax.annotate('', xy=(ex, ey), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#E74C3C',
                                lw=1.5, alpha=alp))

# 高光波瓣弧（窄弧，示意集中分布）
lobe_t = np.linspace(np.radians(22), np.radians(68), 80)
ax.plot(0.72 * np.cos(lobe_t), 0.72 * np.sin(lobe_t),
        '--', color='#E74C3C', lw=1.8, alpha=0.65)

ax.text(0, 2.35, '集中在镜面方向散射\n（高光波瓣）',
        ha='center', fontsize=9.5, color='#E74C3C', style='italic')
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.6, 2.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('镜面反射（Specular Reflection）', fontsize=12, pad=8)

fig.suptitle('图4.1  漫反射与镜面反射对比示意', fontsize=14, y=0.97)
plt.tight_layout()
plt.savefig('docs/pic/ch4_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch4_fig1.png saved")


# ── 图4.2 球谐函数前4阶可视化（极坐标，每阶1个代表性基函数）──────

fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                         subplot_kw=dict(projection='polar'))

theta = np.linspace(0, 2 * np.pi, 720)

def sh_polar_shape(l, theta):
    """返回 (r, pos_mask) — 用轴对称主基函数近似各阶球谐的角度分布"""
    if l == 0:
        val = np.ones_like(theta)
    elif l == 1:
        val = np.cos(theta)           # Y_1^0 主瓣
    elif l == 2:
        val = 3 * np.cos(theta)**2 - 1  # Y_2^0
    else:                             # l == 3
        val = 5 * np.cos(theta)**3 - 3 * np.cos(theta)  # Y_3^0
    r = np.abs(val)
    mx = r.max()
    if mx > 0:
        r = r / mx
    return r, val >= 0

colors_pos = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
colors_neg = '#FF5252'
labels_l = [
    'l = 0（1 个系数）',
    'l = 1（3 个系数）',
    'l = 2（5 个系数）',
    'l = 3（7 个系数）',
]
coef_desc = [
    '仅常数项，颜色与方向无关',
    '捕捉低频方向变化（一阶）',
    '捕捉中频方向变化（二阶）',
    '捕捉更细方向细节（三阶）',
]

for idx, (ax, cp, lbl, cdesc) in enumerate(
        zip(axes.flat, colors_pos, labels_l, coef_desc)):
    l = idx
    r, pos_mask = sh_polar_shape(l, theta)

    r_pos = r.copy(); r_pos[~pos_mask] = 0
    r_neg = r.copy(); r_neg[pos_mask] = 0

    ax.fill(theta, r_pos, color=cp, alpha=0.55)
    ax.plot(theta, r_pos, color=cp, lw=1.8)
    if r_neg.max() > 1e-6:
        ax.fill(theta, r_neg, color=colors_neg, alpha=0.45)
        ax.plot(theta, r_neg, color=colors_neg, lw=1.8)

    ax.set_rticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], fontsize=7.5)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
        fontsize=8)
    ax.set_title(lbl, fontsize=11, pad=14)
    ax.text(0.5, -0.11, cdesc, transform=ax.transAxes,
            ha='center', fontsize=9, color='#555')
    # 图例色块
    ax.text(0.98, 0.98, '正值', transform=ax.transAxes,
            fontsize=8.5, color=cp, ha='right', va='top', fontweight='bold')
    if r_neg.max() > 1e-6:
        ax.text(0.98, 0.87, '负值', transform=ax.transAxes,
                fontsize=8.5, color=colors_neg, ha='right', va='top', fontweight='bold')

fig.suptitle('图4.2  球谐函数前4阶角度分布（极坐标示意，l = 0 ～ 3）',
             fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('docs/pic/ch4_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch4_fig2.png saved")


# ── 图4.3 球谐阶数对颜色拟合的影响 ─────────────────────────────

theta_deg = np.linspace(0, 360, 720)
theta_rad = np.radians(theta_deg)

# 真实目标：包含高频分量的复杂颜色曲线
raw = (0.5
       + 0.28 * np.cos(theta_rad)
       + 0.18 * np.cos(2 * theta_rad)
       + 0.12 * np.sin(3 * theta_rad)
       + 0.16 * np.cos(5 * theta_rad)
       + 0.09 * np.sin(7 * theta_rad)
       + 0.05 * np.cos(11 * theta_rad)
       + 0.03 * np.sin(13 * theta_rad))
target = (raw - raw.min()) / (raw.max() - raw.min()) * 0.82 + 0.06

def sh_fit(y, t, max_l):
    """用 cos/sin 基函数近似 SH 投影拟合（每阶用 m=0 主基函数）"""
    fitted = np.full_like(t, np.mean(y))   # l=0 直流分量
    for l in range(1, max_l + 1):
        c = 2 * np.mean(y * np.cos(l * t))
        s = 2 * np.mean(y * np.sin(l * t))
        fitted = fitted + c * np.cos(l * t) + s * np.sin(l * t)
    return fitted

fit_l0 = sh_fit(target, theta_rad, 0)
fit_l1 = sh_fit(target, theta_rad, 1)
fit_l3 = sh_fit(target, theta_rad, 3)

mse_l0 = np.mean((target - fit_l0)**2)
mse_l1 = np.mean((target - fit_l1)**2)
mse_l3 = np.mean((target - fit_l3)**2)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(theta_deg, target, color='#2C3E50', lw=2.5,
        label='真实颜色（含高频细节）', zorder=5)
ax.plot(theta_deg, fit_l0, color='#E74C3C', lw=2.2, linestyle='--',
        label=f'0 阶拟合（常数），MSE={mse_l0:.4f}', zorder=4)
ax.plot(theta_deg, fit_l1, color='#F39C12', lw=2.2, linestyle='-.',
        label=f'1 阶拟合（低频），MSE={mse_l1:.4f}', zorder=3)
ax.plot(theta_deg, fit_l3, color='#27AE60', lw=2.2, linestyle=':',
        label=f'3 阶拟合（中频），MSE={mse_l3:.4f}', zorder=2)

ax.fill_between(theta_deg, fit_l3, target,
                alpha=0.08, color='#27AE60', label='3 阶拟合误差区域')

ax.set_xlabel('观察角度（度）', fontsize=12)
ax.set_ylabel('颜色亮度（归一化）', fontsize=12)
ax.set_title('图4.3  球谐阶数对颜色拟合的影响', fontsize=13)
ax.set_xlim(0, 360)
ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
ax.set_xticklabels(['0°', '60°', '120°', '180°', '240°', '300°', '360°'])
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

# 简短标注
ax.text(5, float(fit_l0[0]) + 0.04, '0 阶：仅均值',
        fontsize=8.5, color='#E74C3C', style='italic')
ax.text(5, float(fit_l1[0]) - 0.07, '1 阶：低频方向',
        fontsize=8.5, color='#F39C12', style='italic')
ax.text(5, float(fit_l3[15]) + 0.05, '3 阶：中频方向',
        fontsize=8.5, color='#27AE60', style='italic')

note = ('阶数越高，捕捉频率越高\n'
        '但系数数量呈 (l+1)² 增长\n'
        '3DGS 通常使用 l ≤ 3')
ax.text(0.985, 0.04, note, transform=ax.transAxes,
        fontsize=9, color='#555', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                  edgecolor='#CCC', alpha=0.88))

plt.tight_layout()
plt.savefig('docs/pic/ch4_fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("ch4_fig3.png saved")
print("ch4 all done")
