import pandas as pd

# 定义文件路径
file_path = './data/anime_info/anime_data.csv'

# 读取 CSV 文件
df = pd.read_csv(file_path)

# 检查是否有缺失值
print("检查缺失值：")
print(df.isna().sum())

# 删除包含 NaN 的行
df_cleaned = df.dropna()

# 输出清理后的数据
print("\n清理后的数据：")
print(df_cleaned)

# 保存清理后的数据到原文件（覆盖）
df_cleaned.to_csv(file_path, index=False)

print(f"\n清理后的数据已保存到: {file_path}")
