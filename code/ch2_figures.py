"""
Chapter 2 Figures — 3D Gaussian Splatting Book
Generates 5 figures for Chapter 2 covering camera models, coordinate transforms,
distortion, SfM feature matching, and sparse point clouds.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

# ---------------------------------------------------------------------------
# Font setup — Noto Sans CJK SC
# ---------------------------------------------------------------------------
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = FontProperties(fname=FONT_PATH)

# Use FontProperties(fname=...) for all text calls (rcParams name lookup fails for
# TTC collections on this system — the file is read directly via fp instead).
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "/home/root123/Documents/3dgs/docs/pic"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — 针孔相机模型示意
# ---------------------------------------------------------------------------
def fig1_pinhole_camera():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-3, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Optical axis (horizontal center line)
    ax.annotate("", xy=(7.5, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(7.6, 0.1, "光轴", fontproperties=fp, fontsize=11, color="gray")

    # Image plane (left side)
    img_plane_x = 1.5
    ax.plot([img_plane_x, img_plane_x], [-2.5, 2.5], color="steelblue", lw=2.5)
    ax.text(img_plane_x - 0.05, 2.7, "像平面", fontproperties=fp, fontsize=11,
            color="steelblue", ha="center")

    # Pinhole / optical center
    pinhole_x = 3.0
    ax.plot(pinhole_x, 0, "ko", markersize=8, zorder=5)
    ax.text(pinhole_x - 0.05, -0.45, "光心（小孔）", fontproperties=fp,
            fontsize=11, ha="center")

    # Scene point P (right side)
    P_x, P_y = 6.5, 2.2
    ax.plot(P_x, P_y, "r*", markersize=14, zorder=5)
    ax.text(P_x + 0.15, P_y + 0.1, "场景点 $P$", fontproperties=fp,
            fontsize=12, color="darkred")

    # Image point p on image plane (inverted via similar triangles)
    f_dist = pinhole_x - img_plane_x
    scene_depth = P_x - pinhole_x
    p_y = -P_y * f_dist / scene_depth
    p_x = img_plane_x

    ax.plot(p_x, p_y, "bs", markersize=10, zorder=5)
    ax.text(p_x - 0.6, p_y - 0.3, "成像点 $p$", fontproperties=fp,
            fontsize=11, color="steelblue")

    # Projection lines (dashed)
    ax.plot([P_x, pinhole_x], [P_y, 0], "r--", lw=1.3, alpha=0.8)
    ax.plot([pinhole_x, p_x], [0, p_y], "b--", lw=1.3, alpha=0.8)

    # Focal length annotation
    ax.annotate("", xy=(img_plane_x, -2.8), xytext=(pinhole_x, -2.8),
                arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5))
    ax.text((img_plane_x + pinhole_x) / 2, -3.1, "焦距 $f$",
            fontproperties=fp, fontsize=12, color="darkgreen", ha="center")

    # Dashed vertical lines for measurement
    ax.plot([img_plane_x, img_plane_x], [-2.5, -2.8], color="darkgreen",
            lw=1, ls="--", alpha=0.6)
    ax.plot([pinhole_x, pinhole_x], [-0.2, -2.8], color="darkgreen",
            lw=1, ls="--", alpha=0.6)

    ax.set_title("针孔相机模型示意图", fontproperties=fp, fontsize=15, pad=12)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ch2_fig1.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — 四个坐标系变换链（横向流程图）
# ---------------------------------------------------------------------------
def fig2_coordinate_chain():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    boxes = [
        (1.1, 1.75, "世界\n坐标系"),
        (4.0, 1.75, "相机\n坐标系"),
        (7.0, 1.75, "图像\n坐标系"),
        (10.0, 1.75, "像素\n坐标系"),
    ]
    transforms = [
        (2.55, 1.75, "[R, t]", "外参变换"),
        (5.55, 1.75, "[K]",    "内参矩阵"),
        (8.55, 1.75, "像素化",  "量化取整"),
    ]

    box_w, box_h = 1.6, 1.1
    for bx, by, label in boxes:
        rect = patches.FancyBboxPatch(
            (bx - box_w / 2, by - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=2,
            edgecolor="#2c6fad", facecolor="#d6e8f7")
        ax.add_patch(rect)
        ax.text(bx, by, label, fontproperties=fp, fontsize=12,
                ha="center", va="center", color="#1a3a5c")

    arrow_kw = dict(arrowstyle="-|>", color="#e07b39",
                    lw=2.0, mutation_scale=18)
    for tx, ty, top_label, bot_label in transforms:
        ax.annotate("", xy=(tx + 0.45, ty), xytext=(tx - 0.45, ty),
                    arrowprops=arrow_kw)
        ax.text(tx, ty + 0.42, top_label, fontproperties=fp, fontsize=11,
                ha="center", va="bottom", color="#c0392b",
                fontweight="bold")
        ax.text(tx, ty - 0.42, bot_label, fontproperties=fp, fontsize=9,
                ha="center", va="top", color="#555555")

    ax.set_title("四个坐标系变换链", fontproperties=fp, fontsize=15, pad=8)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ch2_fig2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — 径向畸变效果
# ---------------------------------------------------------------------------
def apply_radial_distortion(xn, yn, k1):
    r2 = xn ** 2 + yn ** 2
    factor = 1 + k1 * r2
    return xn * factor, yn * factor


def fig3_distortion():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    titles = ["无畸变（正常网格）", "桶形畸变（$k_1 < 0$）", "枕形畸变（$k_1 > 0$）"]
    k1_vals = [0.0, -0.4, 0.4]

    N = 9  # grid lines per axis
    coords = np.linspace(-1, 1, N)

    for ax, k1, title in zip(axes, k1_vals, titles):
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontproperties=fp, fontsize=11)

        t = np.linspace(-1, 1, 200)
        # Horizontal lines
        for yval in coords:
            xd, yd = apply_radial_distortion(t, np.full_like(t, yval), k1)
            ax.plot(xd, yd, color="#2c6fad", lw=0.9, alpha=0.85)
        # Vertical lines
        for xval in coords:
            xd, yd = apply_radial_distortion(np.full_like(t, xval), t, k1)
            ax.plot(xd, yd, color="#2c6fad", lw=0.9, alpha=0.85)

        # Reference circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "gray", lw=1.0, ls="--", alpha=0.5)

        if k1 != 0:
            ax.text(0, -1.5, f"$k_1 = {k1}$", fontproperties=fp,
                    fontsize=10, ha="center", color="darkred")

    fig.suptitle("径向畸变效果对比", fontproperties=fp, fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ch2_fig3.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — SfM 特征匹配示意
# ---------------------------------------------------------------------------
def fig4_feature_matching():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Left image frame
    left_rect = patches.FancyBboxPatch(
        (0.3, 0.5), 4.0, 5.0,
        boxstyle="square,pad=0.05", lw=2.5,
        edgecolor="#555", facecolor="#f0f4f8")
    ax.add_patch(left_rect)
    ax.text(2.3, 5.75, "图像 1", fontproperties=fp, fontsize=13,
            ha="center", color="#333")

    # Right image frame
    right_rect = patches.FancyBboxPatch(
        (6.7, 0.5), 4.0, 5.0,
        boxstyle="square,pad=0.05", lw=2.5,
        edgecolor="#555", facecolor="#f0f4f8")
    ax.add_patch(right_rect)
    ax.text(8.7, 5.75, "图像 2", fontproperties=fp, fontsize=13,
            ha="center", color="#333")

    # 5 matching point pairs
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    left_pts = np.array([
        [1.1, 4.2], [2.5, 3.5], [1.8, 2.1], [3.2, 1.5], [2.0, 4.8]
    ])
    right_pts = np.array([
        [7.4, 3.9], [8.7, 3.2], [7.9, 1.8], [9.5, 1.3], [8.1, 4.5]
    ])

    # Background unmatched feature points
    for _ in range(12):
        x = np.random.uniform(0.5, 4.1)
        y = np.random.uniform(0.7, 5.2)
        ax.plot(x, y, "o", color="#aaa", markersize=5, alpha=0.5)
    for _ in range(12):
        x = np.random.uniform(7.0, 10.5)
        y = np.random.uniform(0.7, 5.2)
        ax.plot(x, y, "o", color="#aaa", markersize=5, alpha=0.5)

    # Draw matching lines and highlighted points
    for lp, rp, c in zip(left_pts, right_pts, colors):
        ax.plot([lp[0], rp[0]], [lp[1], rp[1]],
                color=c, lw=1.6, alpha=0.75, zorder=2)
        ax.plot(lp[0], lp[1], "o", color=c, markersize=9, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(rp[0], rp[1], "o", color=c, markersize=9, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)

    # Labels
    ax.text(1.1, 4.55, "特征点", fontproperties=fp, fontsize=9,
            color="#e74c3c", ha="center")
    ax.text(5.5, 3.0, "匹配\n对应关系", fontproperties=fp, fontsize=10,
            ha="center", va="center", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="#ccc", alpha=0.9))

    ax.set_title("SfM 特征点匹配示意", fontproperties=fp, fontsize=15, pad=10)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ch2_fig4.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — 稀疏点云示意（3D 散点图）
# ---------------------------------------------------------------------------
def fig5_sparse_pointcloud():
    np.random.seed(7)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Room-like sparse point cloud
    n_floor = 60
    floor_x = np.random.uniform(-3, 3, n_floor)
    floor_y = np.random.uniform(-3, 3, n_floor)
    floor_z = np.random.normal(0, 0.08, n_floor)

    n_wall = 40
    wall_x = np.random.uniform(-3, 3, n_wall)
    wall_y = np.random.uniform(0, 2.5, n_wall)
    wall_z = np.random.normal(3, 0.08, n_wall)

    swall_x = np.random.normal(3, 0.08, n_wall)
    swall_y = np.random.uniform(-3, 3, n_wall)
    swall_z = np.random.uniform(0, 2.5, n_wall)

    n_ceil = 30
    ceil_x = np.random.uniform(-3, 3, n_ceil)
    ceil_y = np.random.uniform(-3, 3, n_ceil)
    ceil_z = np.random.normal(2.5, 0.08, n_ceil)

    ax.scatter(floor_x, floor_y, floor_z, c="#3498db", s=8, alpha=0.7,
               label="地面点")
    ax.scatter(wall_x, wall_y, wall_z, c="#e74c3c", s=8, alpha=0.7,
               label="墙面点")
    ax.scatter(swall_x, swall_y, swall_z, c="#2ecc71", s=8, alpha=0.7,
               label="侧墙点")
    ax.scatter(ceil_x, ceil_y, ceil_z, c="#9b59b6", s=8, alpha=0.7,
               label="天花板点")

    # Camera poses as small pyramids
    cam_positions = [
        (0, -4.5, 1.2),
        (-2.5, -4.0, 1.2),
        (2.5, -4.0, 1.2),
    ]
    for cx, cy, cz in cam_positions:
        ax.scatter([cx], [cy], [cz], c="orange", s=60, marker="^",
                   zorder=5, depthshade=False)
        half = 0.5
        depth = 1.2
        corners = [
            (cx - half, cy + depth, cz + half),
            (cx + half, cy + depth, cz + half),
            (cx - half, cy + depth, cz - half),
            (cx + half, cy + depth, cz - half),
        ]
        for corner in corners:
            ax.plot([cx, corner[0]], [cy, corner[1]], [cz, corner[2]],
                    color="orange", lw=0.8, alpha=0.7)
        face_x = [c[0] for c in corners] + [corners[0][0]]
        face_y = [c[1] for c in corners] + [corners[0][1]]
        face_z = [c[2] for c in corners] + [corners[0][2]]
        ax.plot(face_x, face_y, face_z, color="orange", lw=0.8, alpha=0.7)

    cam_patch = mpatches.Patch(color="orange", label="相机位姿")

    ax.set_xlabel("X", fontproperties=fp, fontsize=10)
    ax.set_ylabel("Y", fontproperties=fp, fontsize=10)
    ax.set_zlabel("Z", fontproperties=fp, fontsize=10)
    ax.set_title("SfM 稀疏点云示意", fontproperties=fp, fontsize=14, pad=12)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(cam_patch)
    labels.append("相机位姿")
    ax.legend(handles, labels, prop=fp, fontsize=9, loc="upper left",
              framealpha=0.8)

    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ch2_fig5.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Chapter 2 figures...")
    fig1_pinhole_camera()
    fig2_coordinate_chain()
    fig3_distortion()
    fig4_feature_matching()
    fig5_sparse_pointcloud()
    print("All 5 figures saved to", OUTPUT_DIR)
