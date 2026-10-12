import matplotlib.pyplot as plt

from src.data import DIABETES_LABELS, load_diabetes_df
from src.eda import describe, plot_correlation_heatmap, plot_distributions, save_figure


def test_describe(toy_df):
    stats = describe(toy_df)
    assert list(stats.index) == list(toy_df.columns)
    for col in ["missing", "count", "mean", "std", "median", "skew", "kurtosis"]:
        assert col in stats.columns
    assert (stats["count"] == len(toy_df)).all()
    assert (stats["missing"] == 0).all()


def test_plot_distributions_has_one_axis_per_variable_plus_hidden_slots():
    df = load_diabetes_df()
    fig = plot_distributions(df, DIABETES_LABELS)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == df.shape[1]


def test_plot_correlation_heatmap(toy_df):
    fig = plot_correlation_heatmap(toy_df)
    heatmap_ax = fig.axes[0]
    assert len(heatmap_ax.texts) == toy_df.shape[1] ** 2  # 每个格子一个数值标注
    assert heatmap_ax.images[0].get_clim() == (-1, 1)


def test_save_figure_creates_file_and_closes(tmp_path, toy_df):
    fig = plot_correlation_heatmap(toy_df)
    out = save_figure(fig, tmp_path / "nested" / "heatmap.png")
    assert out.exists() and out.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)
