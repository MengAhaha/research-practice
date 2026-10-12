"""对 sklearn 糖尿病数据集做探索性分析，输出图表和统计表。

在项目根目录运行：python -m src.run_eda
"""

from pathlib import Path

from src.data import DIABETES_LABELS, load_diabetes_df
from src.eda import describe, plot_correlation_heatmap, plot_distributions, save_figure

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"


def main() -> None:
    df = load_diabetes_df()
    print(f"数据：{df.shape[0]} 行 × {df.shape[1]} 列")

    TABLES.mkdir(parents=True, exist_ok=True)
    stats = describe(df)
    stats.round(4).to_csv(TABLES / "diabetes_describe.csv", index_label="variable")
    df.corr().round(4).to_csv(TABLES / "diabetes_correlation.csv", index_label="variable")

    outputs = [
        save_figure(plot_distributions(df, DIABETES_LABELS), FIGURES / "diabetes_distributions.png"),
        save_figure(plot_correlation_heatmap(df, DIABETES_LABELS), FIGURES / "diabetes_correlation_heatmap.png"),
        TABLES / "diabetes_describe.csv",
        TABLES / "diabetes_correlation.csv",
    ]
    for p in outputs:
        print(f"已保存：{p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
