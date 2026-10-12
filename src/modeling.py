"""建模：线性回归与随机森林的交叉验证对比，以及变量重要性。"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_LABELS = {"linear": "线性回归", "forest": "随机森林"}


def split_xy(df: pd.DataFrame, target: str = "target") -> tuple[pd.DataFrame, pd.Series]:
    """把 DataFrame 拆成特征 X 和目标 y。"""
    return df.drop(columns=[target]), df[target]


def make_linear() -> object:
    """线性回归。先标准化，使各变量的系数可以直接比较大小。"""
    return make_pipeline(StandardScaler(), LinearRegression())


def make_forest(random_state: int = RANDOM_STATE, n_estimators: int = 300) -> RandomForestRegressor:
    """随机森林，使用默认超参数（不针对本数据集调参）。"""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def make_models(random_state: int = RANDOM_STATE, n_estimators: int = 300) -> dict:
    return {"linear": make_linear(), "forest": make_forest(random_state, n_estimators)}


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict | None = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """用同一组 K 折划分比较多个模型。

    返回 (每折得分, 汇总, 交叉验证预测值)：
    - 每折得分：每行一个 模型×折，含 r2 / rmse / mae
    - 汇总：每个模型一行，含各指标的 mean 和 std
    - 交叉验证预测值：每个样本由"没见过它"的那一折模型预测，每列一个模型
    """
    models = models if models is not None else make_models(random_state)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    rows, preds = [], {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        for k in range(n_splits):
            rows.append(
                {
                    "model": name,
                    "fold": k + 1,
                    "r2": res["test_r2"][k],
                    "rmse": -res["test_rmse"][k],
                    "mae": -res["test_mae"][k],
                }
            )
        preds[name] = cross_val_predict(model, X, y, cv=cv)

    folds = pd.DataFrame(rows)
    summary = folds.groupby("model", sort=False)[["r2", "rmse", "mae"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return folds, summary, pd.DataFrame(preds, index=y.index)


def linear_coefficients(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """在全部数据上拟合线性回归，返回每个变量的系数。

    - coef_std：标准化系数，变量每增加 1 个标准差，目标的变化量（变量间可比）
    - coef_raw：原始单位系数，变量每增加 1 个单位，目标的变化量
    """
    model = make_linear().fit(X, y)
    scaler, reg = model.named_steps["standardscaler"], model.named_steps["linearregression"]
    return pd.DataFrame(
        {"coef_std": reg.coef_, "coef_raw": reg.coef_ / scaler.scale_}, index=X.columns
    )


def forest_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 300,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """随机森林的变量重要性。

    - impurity：全部数据上拟合得到的基尼/方差减少量重要性（偏向取值多的连续变量）
    - permutation_mean / permutation_std：在每一折的验证集上打乱该变量后 R² 的下降量，
      再对各折取均值和标准差。衡量的是变量对"没见过的数据"的预测贡献，更可靠。
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_fold = []
    for train, test in cv.split(X):
        rf = make_forest(random_state, n_estimators).fit(X.iloc[train], y.iloc[train])
        res = permutation_importance(
            rf, X.iloc[test], y.iloc[test], n_repeats=n_repeats, random_state=random_state
        )
        per_fold.append(res.importances_mean)
    per_fold = np.array(per_fold)

    full = make_forest(random_state, n_estimators).fit(X, y)
    return pd.DataFrame(
        {
            "impurity": full.feature_importances_,
            "permutation_mean": per_fold.mean(axis=0),
            "permutation_std": per_fold.std(axis=0),
        },
        index=X.columns,
    )


def combine_importance(
    coefs: pd.DataFrame, forest: pd.DataFrame, labels: dict | None = None
) -> pd.DataFrame:
    """合并两个模型的结果，给出各自排名和平均排名（按平均排名升序）。

    线性回归按 |标准化系数| 排名，随机森林按置换重要性排名。
    """
    labels = labels or {}
    table = coefs.join(forest)
    table.insert(0, "label", [labels.get(c, c) for c in table.index])
    table["linear_rank"] = table["coef_std"].abs().rank(ascending=False, method="min").astype(int)
    table["forest_rank"] = table["permutation_mean"].rank(ascending=False, method="min").astype(int)
    table["avg_rank"] = (table["linear_rank"] + table["forest_rank"]) / 2
    table.index.name = "variable"
    return table.sort_values(["avg_rank", "forest_rank"])
