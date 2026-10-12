import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    combine_importance,
    cross_validate_models,
    forest_importance,
    linear_coefficients,
    make_models,
    split_xy,
)

FAST = dict(n_estimators=30)


@pytest.fixture(scope="module")
def xy(toy_df):
    return split_xy(toy_df)


def test_split_xy(toy_df):
    X, y = split_xy(toy_df)
    assert "target" not in X.columns and list(X.columns) == ["x0", "x1", "x2", "x3"]
    assert y.name == "target" and len(y) == len(X)


def test_cross_validate_models_shapes_and_scores(xy):
    X, y = xy
    folds, summary, preds = cross_validate_models(X, y, models=make_models(n_estimators=30), n_splits=5)
    assert len(folds) == 2 * 5
    assert set(folds["model"]) == {"linear", "forest"}
    assert list(summary.index) == ["linear", "forest"]
    assert {"r2_mean", "r2_std", "rmse_mean", "mae_mean"} <= set(summary.columns)
    assert preds.shape == (len(y), 2) and preds.index.equals(y.index)
    # 数据近似线性，线性回归应拟合得很好
    assert summary.loc["linear", "r2_mean"] > 0.9
    assert (folds[["rmse", "mae"]] >= 0).all().all()


def test_cross_validate_models_is_reproducible(xy):
    X, y = xy
    kw = dict(models=make_models(n_estimators=20), n_splits=3)
    a = cross_validate_models(X, y, **kw)[1]
    b = cross_validate_models(X, y, **kw)[1]
    pd.testing.assert_frame_equal(a, b)


def test_linear_coefficients_recover_signs_and_ranking(xy):
    X, y = xy
    coefs = linear_coefficients(X, y)
    assert list(coefs.columns) == ["coef_std", "coef_raw"]
    assert coefs.loc["x0", "coef_raw"] == pytest.approx(3, abs=0.3)
    assert coefs.loc["x1", "coef_raw"] == pytest.approx(-2, abs=0.3)
    assert coefs["coef_std"].abs().idxmax() == "x0"
    # 标准化系数 = 原始系数 × 该变量的标准差
    assert coefs.loc["x0", "coef_std"] == pytest.approx(coefs.loc["x0", "coef_raw"] * X["x0"].std(ddof=0))


def test_forest_importance(xy):
    X, y = xy
    imp = forest_importance(X, y, n_splits=3, n_repeats=3, **FAST)
    assert list(imp.index) == list(X.columns)
    assert imp["impurity"].sum() == pytest.approx(1.0)
    assert imp["permutation_mean"].idxmax() == "x0"
    assert imp.loc[["x0", "x1"], "permutation_mean"].min() > imp.loc[["x2", "x3"], "permutation_mean"].max()


def test_combine_importance_ranks(xy):
    X, y = xy
    table = combine_importance(
        linear_coefficients(X, y), forest_importance(X, y, n_splits=3, n_repeats=3, **FAST), {"x0": "变量0"}
    )
    assert table.index.name == "variable"
    assert table.index[0] == "x0" and table.loc["x0", "label"] == "变量0"
    assert table.loc["x2", "label"] == "x2"  # 没有中文名时回退到变量名
    assert table["avg_rank"].is_monotonic_increasing
    assert set(table["linear_rank"]) <= set(range(1, 5))
    assert np.allclose(table["avg_rank"], (table["linear_rank"] + table["forest_rank"]) / 2)
