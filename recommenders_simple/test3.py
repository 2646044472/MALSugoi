import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def convert_to_int(rating_str):
    try:
        return int(rating_str)
    except ValueError:
        print(f"Warning: Cannot convert '{rating_str}' to an integer. Setting to 0.")
        return 0

def df_to_sparse_matrix(df):
    data = []
    users = set()
    animes = set()

    for index, row in df.iterrows():
        user, anime, rating = row['username'], row['anime'], row['rating']
        if rating == '-':
            rating = 0
        rating = convert_to_int(rating)
        data.append((user, anime, rating))
        users.add(user)
        animes.add(anime)

    users = sorted(list(users))
    animes = sorted(list(animes))

    # 创建稀疏矩阵
    rows = []
    cols = []
    values = []

    for user, anime, rating in data:
        if rating > 0:  # 仅处理正评分
            rows.append(animes.index(anime))
            cols.append(users.index(user))
            values.append(rating)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(animes), len(users)))

    return sparse_matrix, users, animes  # 返回稀疏矩阵

def process_sparse_matrix(sparse_matrix):
    # 计算每行的总和和非零元素的计数
    row_sum = sparse_matrix.sum(axis=1).A1  # 转换为一维数组
    row_count = (sparse_matrix > 0).sum(axis=1).A1  # 转换为一维数组

    # 使用广播计算每行的平均值
    row_mean = np.where(row_count > 0, row_sum / row_count, 1)  # 避免除以零

    # 归一化稀疏矩阵
    normalized_matrix = sparse_matrix.copy()
    for i in range(normalized_matrix.shape[0]):
        if row_mean[i] != 0:  # 确保不除以零
            normalized_matrix[i, :] /= row_mean[i]

    return normalized_matrix

def calculate_row_cosine_similarity(sparse_matrix):
    # 使用 sklearn 的 cosine_similarity 函数直接计算稀疏矩阵的相似度
    similarity_matrix = cosine_similarity(sparse_matrix)
    return pd.DataFrame(similarity_matrix)

# 示例用法
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/data/user_animelist/anime_info.csv'
df = pd.read_csv(csv_file_path)

# 步骤1: 将 DataFrame 转换为稀疏矩阵
sparse_matrix, users, animes = df_to_sparse_matrix(df)

# 步骤2: 处理稀疏矩阵
processed_sparse_matrix = process_sparse_matrix(sparse_matrix)

# 步骤3: 计算行之间的余弦相似度
row_similarity = calculate_row_cosine_similarity(processed_sparse_matrix)

# 打印相似度矩阵
print(row_similarity)