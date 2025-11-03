import pandas as pd
from scipy.stats import pearsonr, spearmanr

x = [1,2,3,4,5,6]
y = [2,4,5,7,10,15]

# 计算皮尔逊相关系数
pearson_corr, pearson_p = pearsonr(x, y)
print(f"皮尔逊相关系数: {pearson_corr:.4f}, p值: {pearson_p:.4f}")

# 计算斯皮尔曼相关系数
spearman_corr, spearman_p = spearmanr(x, y)
print(f"斯皮尔曼相关系数: {spearman_corr:.4f}, p值: {spearman_p:.4f}")

# 显示原始数据
print(f"\n原始数据:")
print(f"x: {x}")
print(f"y: {y}")
