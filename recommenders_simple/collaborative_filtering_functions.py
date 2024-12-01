import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
def parallel_cosine_similarity(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows), dtype=float)

    def calculate_similarity(i):
        row_a = matrix.iloc[i, :].values.copy()  # 确保是可写的
        row_similarities = []
        for j in range(num_rows):
            row_b = matrix.iloc[j, :].values.copy()
            sim_value = cosine(row_a, row_b)
            row_similarities.append(sim_value)
        return row_similarities

    # 使用并行计算获取每一行的相似度
    results = Parallel(n_jobs=-1)(delayed(calculate_similarity)(i) for i in range(num_rows))

    # 填充相似度矩阵
    for i in range(num_rows):
        for j in range(num_rows):
            similarity_matrix[i, j] = results[i][j]

    return pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)
def convert_to_int(rating_str):
    try:
        # 尝试将字符串转换为整数
        return int(rating_str)
    except ValueError:
        # 如果转换失败，返回0或其他默认值
        print(f"Warning: Cannot convert '{rating_str}' to an integer. Setting to 0.")
        return 0
def check_non_integer_entries(matrix):
    # 遍历矩阵的每一行和每一列
    for index, row in matrix.iterrows():
        for col in matrix.columns:
            value = row[col]
            if not np.issubdtype(type(value), np.integer) and not np.issubdtype(type(value), np.floating):
                print(f"Non-integer entry found at (username: {col}, anime: {index}): {value}")

def cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 如果其中一个向量的范数为0，返回相似度为0（可以根据需求调整）
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return cos_sim

def df_to_matrix2(df):
    return df.pivot(index='username', columns='anime', values='rating').fillna(0)
def df_to_matrix(df):
    data = []
    users = set()
    animes = set()

    for index, row in df.iterrows():
        user, anime, rating = row['username'], row['anime'], row['rating']
        # 将评分为 '-' 的情况直接设置为 0
        if rating == '-':
            rating = 0  # 将字符串 '0' 传递给 convert_to_int
        # 使用 convert_to_int 函数转换评分
        rating = convert_to_int(rating)
        data.append((user, anime, rating))
        users.add(user)
        animes.add(anime)

    users = sorted(list(users))
    animes = sorted(list(animes))

    matrix = pd.DataFrame(np.zeros((len(animes), len(users))), index=animes, columns=users)

    for user, anime, rating in data:
        matrix.at[anime, user] = rating

    return matrix.astype(int)  # 确保所有评分都是整数
def process_matrix(matrix):
    # 创建一个空列表来存储每行的处理结果
    results = []

    # 遍历矩阵的每一行
    for index, row in matrix.iterrows():
        count = 0  # 记录正整数的数量
        row_sum = 0  # 记录这一行正整数的总和

        # 遍历每一行中的每个元素
        for value in row:
            if value > 0:  # 只处理正整数
                count += 1
                row_sum += value

        # 计算 sum / count
        if count > 0:  # 避免除以零
            divide_result = row_sum / count
        else:
            divide_result = 0  # 如果没有正整数，则除以零时返回0

        # 对每个元素进行操作：如果是正整数，除以 divide_result；如果是0，保持不变
        new_row = []
        for value in row:
            if value > 0:
                new_row.append(value / divide_result)  # 除以 result
            else:
                new_row.append(value)  # 如果是0，保持原样

        # 将新的行添加到结果列表
        results.append(new_row)

    # 返回一个新的 DataFrame，包含处理后的矩阵
    result_matrix = pd.DataFrame(results, columns=matrix.columns, index=matrix.index)
    return result_matrix
def print_row_similarity(matrix, row_index1, row_index2):
    # 提取指定的两行
    row1 = matrix.iloc[row_index1, :].values
    row2 = matrix.iloc[row_index2, :].values
    
    # 计算余弦相似度
    similarity = cosine(row1, row2)
    
    # 打印结果
    print(f"Cosine similarity between row {row_index1} and row {row_index2}: {similarity:.4f}")
def calculate_row_cosine_similarity(matrix):
    # 获取矩阵的行数
    num_rows = matrix.shape[0]
    
    # 初始化一个空的相似度矩阵
    similarity_matrix = pd.DataFrame(np.zeros((num_rows, num_rows)),
                                      index=matrix.index,
                                      columns=matrix.index)

    # 遍历每对行并计算余弦相似度
    for i in range(num_rows):
        for j in range(i, num_rows):
            # 提取矩阵中的第 i 行和第 j 行
            row_i = matrix.iloc[i, :].values
            row_j = matrix.iloc[j, :].values
            
            # 计算余弦相似度
            sim = cosine(row_i, row_j)

            # 将相似度存储在相似度矩阵中
            similarity_matrix.iloc[i, j] = sim
            similarity_matrix.iloc[j, i] = sim  # 确保矩阵对称

    return similarity_matrix
# 示例用法
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/data/user_animelist/anime_info.csv'
df = pd.read_csv(csv_file_path)
matrix = df_to_matrix(df)

# 处理矩阵
sim = process_matrix(matrix)
check_non_integer_entries(sim)

# 计算相似度
sim = parallel_cosine_similarity(sim)
print(sim)