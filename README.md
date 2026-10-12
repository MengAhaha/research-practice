# research-practice

一个 Python 科研项目的基本骨架，用于组织数据、代码、探索性分析和实验结果。

## 目录结构

```
research-practice/
├── data/                # 数据
│   ├── raw/             # 原始数据：从外部获取后保持原样，只读不改
│   └── processed/       # 处理后数据：清洗、转换后的中间或最终数据
├── src/                 # 源代码：可复用的函数、模型、数据处理流程
├── notebooks/           # Jupyter 笔记本：探索性分析、可视化、快速实验
├── results/             # 实验结果
│   ├── figures/         # 图表
│   └── tables/          # 表格、统计结果
├── requirements.txt     # Python 依赖列表
└── README.md            # 项目说明
```

### 各目录用途

- **`data/`**：存放项目用到的所有数据。
  - `raw/` 放原始数据，**不要手动修改**，保证任何结果都能从原始数据重新生成。
  - `processed/` 放由 `src/` 中脚本生成的处理后数据。
- **`src/`**：项目的核心代码。数据加载、预处理、特征工程、模型、评估等逻辑都写在这里，
  以便在笔记本和脚本之间复用，也便于测试。
- **`notebooks/`**：用于探索和原型验证。建议用编号加简短描述命名，
  例如 `01-数据探索.ipynb`、`02-基线模型.ipynb`。成熟的代码应整理进 `src/`。
- **`results/`**：存放实验产出，如图表（`figures/`）和结果表格（`tables/`），
  便于撰写论文或报告时直接引用。

> 说明：`data/` 下的文件默认被 `.gitignore` 忽略（只保留空目录占位的 `.gitkeep`），避免把大文件提交到仓库。
> `results/` 下的图表和表格会正常提交，方便直接在 GitHub 上查看。

## 快速开始

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter lab
```

在笔记本中导入 `src` 里的代码（笔记本位于 `notebooks/` 时）：

```python
import sys
sys.path.append("..")

from src import utils  # 将 utils 替换为 src/ 中实际的模块名
```

## 示例：糖尿病数据集探索性分析

使用 sklearn 自带的 `load_diabetes` 数据集（442 名患者，10 个基线特征 + 一年后病情进展指标）：

```bash
python -m src.run_eda
```

| 文件 | 内容 |
|---|---|
| `src/data.py` | `load_diabetes_df()`：加载数据为 DataFrame（默认原始单位）；`DIABETES_LABELS`：变量中文名 |
| `src/eda.py` | `describe()` 描述统计、`plot_distributions()` 分布图、`plot_correlation_heatmap()` 相关性热图 |
| `src/run_eda.py` | 一键运行，生成下列结果 |

输出：

- `results/figures/diabetes_distributions.png`：各变量分布
- `results/figures/diabetes_correlation_heatmap.png`：相关性热图
- `results/tables/diabetes_describe.csv`：描述统计（缺失数、均值、标准差、分位数、偏度、峰度）
- `results/tables/diabetes_correlation.csv`：相关系数矩阵
