# 软件分析与测试论文复现实验

代码仓库地址：https://github.com/Mimlv/Mimlv

数据集 URL：
- KC1: https://www.openml.org/d/1067, download: https://www.openml.org/data/download/53950/kc1.arff
- JM1: https://www.openml.org/d/1053, download: https://www.openml.org/data/download/53936/jm1.arff
- PC1: https://www.openml.org/d/1068, download: https://www.openml.org/data/download/53951/pc1.arff

运行方式：`python run_defect_experiment.py`。脚本会执行项目内模型比较、测试预算敏感性分析、特征组消融、跨项目迁移和特征相关性分析，并输出 CSV 与 PNG 图像。
