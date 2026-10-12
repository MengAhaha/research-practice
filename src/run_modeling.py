"""回答研究问题：哪些变量最能预测疾病进展？

线性回归 vs 随机森林，5 折交叉验证。在项目根目录运行：python -m src.run_modeling
"""

from pathlib import Path

import pandas as pd

from src.data import DIABETES_LABELS, load_diabetes_df
from src.eda import save_figure
from src.model_plots import plot_feature_importance, plot_pred_vs_actual
from src.modeling import (
    MODEL_LABELS,
    combine_importance,
    cross_validate_models,
    forest_importance,
    linear_coefficients,
    split_xy,
)

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"


def main() -> None:
    X, y = split_xy(load_diabetes_df())

    folds, summary, preds = cross_validate_models(X, y)
    ranking = combine_importance(
        linear_coefficients(X, y), forest_importance(X, y), DIABETES_LABELS
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    folds.round(4).to_csv(TABLES / "cv_scores_by_fold.csv", index=False)
    summary.round(4).to_csv(TABLES / "cv_scores_summary.csv", index_label="model")
    ranking.round(4).to_csv(TABLES / "feature_ranking.csv", index_label="variable")
    save_figure(plot_feature_importance(ranking), FIGURES / "feature_importance.png")
    save_figure(plot_pred_vs_actual(y, preds, summary), FIGURES / "pred_vs_actual.png")

    shown = summary.rename(index=MODEL_LABELS).round(3)
    print("5 折交叉验证结果：")
    print(shown.to_string())
    print("\n变量排名：")
    cols = ["label", "coef_std", "linear_rank", "permutation_mean", "forest_rank", "avg_rank"]
    print(ranking[cols].round(3).to_string())
    print("\n已保存到 results/figures/ 和 results/tables/")


if __name__ == "__main__":
    main()
