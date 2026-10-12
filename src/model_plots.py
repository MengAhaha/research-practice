"""建模结果可视化：特征重要性图、预测值-真实值散点图。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import BASELINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES, SURFACE, TARGET
from src.modeling import MODEL_LABELS

# 每个模型固定一种颜色，两张图里保持一致
MODEL_COLORS = {"linear": SERIES, "forest": TARGET}


def _barh(ax, names, values, color, errors=None):
    """画一个降序的水平条形图，条形末端直接标数值。"""
    y = np.arange(len(names))
    ax.barh(y, values, height=0.62, color=color, edgecolor=SURFACE, linewidth=1)
    if errors is not None:
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=2)
    span = max(abs(values).max(), 1e-9)
    pad = span * 0.02
    for yi, v in zip(y, values):
        err = 0 if errors is None else errors[list(y).index(yi)]
        offset = pad + err
        ax.text(
            v + offset if v >= 0 else v - offset, yi, f"{v:.3f}" if span < 10 else f"{v:.1f}",
            ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK_SECONDARY,
        )
    ax.set_yticks(y, names, fontsize=9, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.grid(True, axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=8)
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    ax.set_xlim(lo - span * 0.18 if lo < 0 else 0, hi + span * 0.18)


def plot_feature_importance(table: pd.DataFrame) -> plt.Figure:
    """左：线性回归标准化系数；右：随机森林置换重要性。各自按重要性降序。

    table 为 modeling.combine_importance 的输出。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.5 * len(table) + 2.2))

    lin = table.reindex(table["coef_std"].abs().sort_values(ascending=False).index)
    _barh(ax1, [f"{r.label} ({i})" for i, r in lin.iterrows()], lin["coef_std"].to_numpy(), SERIES)
    ax1.set_title(f"{MODEL_LABELS['linear']}：标准化系数", fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("变量每增加 1 个标准差，病情进展指标的变化量（正负表示方向）", fontsize=9)

    rf = table.sort_values("permutation_mean", ascending=False)
    _barh(
        ax2, [f"{r.label} ({i})" for i, r in rf.iterrows()], rf["permutation_mean"].to_numpy(),
        TARGET, errors=rf["permutation_std"].to_numpy(),
    )
    ax2.set_title(f"{MODEL_LABELS['forest']}：置换重要性", fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("打乱该变量后验证集 R² 的下降量（误差线为 5 折标准差）", fontsize=9)

    fig.suptitle("哪些变量最能预测病情进展", fontsize=13, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y: pd.Series, preds: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    """各模型的交叉验证预测值 vs 真实值散点图，虚线为 y = x（完美预测）。"""
    models = list(preds.columns)
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo = float(min(y.min(), preds.min().min()))
    hi = float(max(y.max(), preds.max().max()))
    pad = (hi - lo) * 0.04

    for ax, name in zip(axes, models):
        color = MODEL_COLORS.get(name, SERIES)
        ax.scatter(y, preds[name], s=26, color=color, alpha=0.7, edgecolor=SURFACE, linewidth=0.6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_SECONDARY, linewidth=1, linestyle="--")
        s = summary.loc[name]
        ax.set_title(
            f"{MODEL_LABELS.get(name, name)}\n"
            f"5 折平均 R² = {s['r2_mean']:.2f} ± {s['r2_std']:.2f}，RMSE = {s['rmse_mean']:.1f}",
            fontsize=10, loc="left",
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(True, axis="both")
        ax.tick_params(length=0, labelsize=8)
        ax.set_xlabel("真实值（一年后病情进展）", fontsize=9)
    axes[0].set_ylabel("交叉验证预测值", fontsize=9)

    fig.suptitle("预测值 vs 真实值（每个样本由未见过它的那一折模型预测）",
                 fontsize=13, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    return fig
