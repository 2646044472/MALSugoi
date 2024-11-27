import pandas as pd
import numpy as np
from typing import Tuple

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def df_to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 转换为评分矩阵"""
    # 初始化列表以存储提取的用户、动漫和评分
    data = []
    users = set()
    animes = set()

    # 遍历 DataFrame 并提取相关信息
    for _, row in df.iterrows():
        user, anime, rating = row['username'], row['anime'], row['rating']
        # 处理缺失评分（'-'）
        rating = 0 if rating == '-' else int(rating)  # 将 '-' 替换为 0
        data.append((user, anime, rating))
        users.add(user)
        animes.add(anime)

    # 将用户和动漫转换为排序列表（以保持一致的列/行顺序）
    users = sorted(users)
    animes = sorted(animes)

    # 创建一个空的 DataFrame，用户为列，动漫标题为行
    matrix = pd.DataFrame(0, index=animes, columns=users)  # 初始化为 0

    # 填充矩阵
    for user, anime, rating in data:
        matrix.at[anime, user] = rating

    return matrix

# 示例用法：读取 CSV 文件到 DataFrame
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/anime_info.csv'  # 调整文件路径
df = pd.read_csv(csv_file_path)
print(df)
matrix = df_to_matrix(df)
print(matrix)