"""数据加载。"""

import pandas as pd
from sklearn.datasets import load_diabetes

# 各变量的中文含义（来自 sklearn 数据集说明）
DIABETES_LABELS = {
    "age": "年龄",
    "sex": "性别",
    "bmi": "体重指数 BMI",
    "bp": "平均血压",
    "s1": "总胆固醇 TC",
    "s2": "低密度脂蛋白 LDL",
    "s3": "高密度脂蛋白 HDL",
    "s4": "总胆固醇/HDL",
    "s5": "甘油三酯（对数）",
    "s6": "血糖",
    "target": "一年后病情进展",
}


def load_diabetes_df(scaled: bool = False) -> pd.DataFrame:
    """加载 sklearn 糖尿病数据集，返回包含 10 个特征和 target 列的 DataFrame。

    scaled=False 时返回原始单位（年龄为岁、BMI 为 kg/m² 等），便于解读；
    scaled=True 时返回 sklearn 默认的中心化、标准化后数值。
    """
    return load_diabetes(as_frame=True, scaled=scaled).frame
