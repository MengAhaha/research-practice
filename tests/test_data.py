from src.data import DIABETES_LABELS, load_diabetes_df


def test_load_shape_and_columns():
    df = load_diabetes_df()
    assert df.shape == (442, 11)
    assert list(df.columns) == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target"]
    assert not df.isna().any().any()


def test_raw_units_vs_scaled():
    raw, scaled = load_diabetes_df(), load_diabetes_df(scaled=True)
    assert raw["age"].between(19, 79).all()  # 原始单位：岁
    assert abs(scaled["age"].mean()) < 1e-6  # 标准化后均值为 0
    assert raw["target"].equals(scaled["target"])  # 目标变量不变


def test_labels_cover_all_columns():
    assert set(DIABETES_LABELS) == set(load_diabetes_df().columns)
