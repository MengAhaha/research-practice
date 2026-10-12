import pytest

from src.eda import save_figure
from src.model_plots import plot_feature_importance, plot_pred_vs_actual
from src.modeling import (
    combine_importance,
    cross_validate_models,
    forest_importance,
    linear_coefficients,
    make_models,
    split_xy,
)


@pytest.fixture(scope="module")
def results(toy_df):
    X, y = split_xy(toy_df)
    _, summary, preds = cross_validate_models(X, y, models=make_models(n_estimators=20), n_splits=3)
    table = combine_importance(
        linear_coefficients(X, y), forest_importance(X, y, n_splits=3, n_repeats=2, n_estimators=20)
    )
    return y, preds, summary, table


def test_plot_feature_importance(results, tmp_path):
    _, _, _, table = results
    fig = plot_feature_importance(table)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.patches) == len(table)  # 每个变量一根条
    assert save_figure(fig, tmp_path / "fi.png").exists()


def test_plot_pred_vs_actual(results, tmp_path):
    y, preds, summary, _ = results
    fig = plot_pred_vs_actual(y, preds, summary)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.collections[0].get_offsets()) == len(y)  # 每个样本一个点
        assert "R²" in ax.get_title(loc="left")
    assert save_figure(fig, tmp_path / "pv.png").exists()
