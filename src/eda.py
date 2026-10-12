"""探索性分析：描述统计与可视化。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# 配色与图表样式
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"  # 单变量分布统一用一种颜色
TARGET = "#eb6834"  # 目标变量用第二种颜色，与特征区分
# 相关系数：蓝（负）- 灰（0）- 红（正）
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"]
)

plt.rcParams.update(
    {
        # 中文字体：按常见系统依次尝试
        "font.sans-serif": [
            "PingFang SC",
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个变量的描述统计量，每行一个变量。"""
    stats = df.describe().T
    stats.insert(0, "missing", df.isna().sum())
    stats["skew"] = df.skew()
    stats["kurtosis"] = df.kurtosis()
    return stats.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})


def plot_distributions(
    df: pd.DataFrame, labels: dict | None = None, target: str = "target", ncols: int = 4
) -> plt.Figure:
    """为每个变量画一个分布小图：连续变量用直方图，取值少的离散变量用条形图。"""
    labels = labels or {}
    cols = list(df.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        color = TARGET if col == target else SERIES
        if s.nunique() <= 5:
            counts = s.value_counts().sort_index()
            ax.bar(
                [f"{v:g}" for v in counts.index], counts.values,
                width=0.5, color=color, edgecolor=SURFACE, linewidth=2,
            )
        else:
            ax.hist(s, bins=25, color=color, edgecolor=SURFACE, linewidth=1)
            ax.axvline(s.mean(), color=INK_SECONDARY, linewidth=1, linestyle="--")
        name = labels.get(col, col)
        ax.set_title(f"{name}（{col}）" if name != col else col, fontsize=10, loc="left")
        ax.tick_params(labelsize=8, length=0)
        ax.set_ylabel("样本数", fontsize=8)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "各变量分布（虚线为均值；橙色为目标变量）",
        fontsize=13, color=INK_PRIMARY, x=0.01, ha="left",
    )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, labels: dict | None = None, method: str = "pearson"
) -> plt.Figure:
    """画变量间相关系数热图，色阶固定在 [-1, 1]，格子内标注数值。"""
    labels = labels or {}
    corr = df.corr(method=method)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.65 * n + 1.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) >= 0.6 else INK_PRIMARY,
            )

    # 格子之间留出表面色细缝
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ticklabels = [f"{labels[c]} ({c})" if c in labels else c for c in corr.columns]
    ax.set_xticks(range(n), ticklabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), ticklabels, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0, colors=INK_MUTED)
    cbar.set_label(f"相关系数（{method}）", color=INK_SECONDARY, fontsize=9)

    ax.set_title("变量相关性热图", fontsize=13, loc="left", pad=12)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """保存图片并关闭 figure，自动创建目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
