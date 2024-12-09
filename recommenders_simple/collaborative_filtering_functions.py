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
import numpy as np
import pandas as pd

def item_based_recommendation(similarity_matrix, matrix, top_n=5000, user_index=None, anime_index=None):
    """
    计算用户对动漫的推荐评分 r_ix。
    
    参数:
    - similarity_matrix: 动漫之间的相似度矩阵 (DataFrame)
    - rating_matrix: 用户对动漫的评分矩阵 (DataFrame)
    - top_n: 考虑的与动漫 i 相似的前 N 个动漫 (int)
    - user_index: 用户的索引 (str)
    - anime_index: 当前动漫的索引 (str)
    
    返回:
    - 推荐评分 r_ix (float)
    """
    
    # 检查 anime_index 是否在 rating_matrix 中
    if anime_index not in matrix.index:
        print(f"动漫 {anime_index} 不在评分矩阵中。")
        return 0  # 或者返回一个合适的默认值

    # 检查 user_index 是否在 rating_matrix 中
    if user_index not in matrix.columns:
        print(f"用户 {user_index} 不在评分矩阵中。")
        return 0  # 或者返回一个合适的默认值

    # 获取用户对该动漫的评分
    user_rating = matrix.loc[anime_index, user_index]  # 使用 .loc

    # 如果评分不为 0，直接返回该评分
    if user_rating != 0:
        return user_rating
    
    # 提取动漫 i 的相似度和用户的评分
    sim_scores = similarity_matrix.loc[anime_index, :].values  # 使用 .loc
    ratings = matrix.loc[:, user_index].values  # 使用 .loc，获取用户的评分

    # 找到与动漫 i 最相似的前 N 个动漫的索引
    similar_indices = np.argsort(sim_scores)[-top_n:][::-1]

    # 计算推荐评分 r_ix
    numerator = 0  # 分子
    denominator = 0  # 分母
    
    for j in similar_indices:
        if ratings[j] > 0:  # 只考虑用户已评分的动漫
            numerator += sim_scores[j] * ratings[j]
            denominator += sim_scores[j]

    # 避免除以零
    if denominator > 0:
        return numerator / denominator
    else:
        return 0  # 如果没有相似度，返回 0
def calculate_all_recommendations(similarity_matrix, rating_matrix, user_index, top_n=100):
    """
    计算用户对所有动漫的推荐评分，并选出前 N 部评分最高的动漫。
    
    参数:
    - similarity_matrix: 动漫之间的相似度矩阵 (DataFrame)
    - rating_matrix: 用户对动漫的评分矩阵 (DataFrame)
    - user_index: 用户的索引 (str)
    - top_n: 返回前 N 部评分最高的动漫 (int)

    返回:
    - 推荐动漫及其评分 (DataFrame)
    """
    n_animes = rating_matrix.shape[0]  # 动漫数量
    recommended_ratings = []

    for anime_index in rating_matrix.index:  # 使用行名（动漫名称）
        rating = item_based_recommendation(similarity_matrix, rating_matrix, user_index=user_index, anime_index=anime_index)
        recommended_ratings.append((anime_index, rating))

    # 创建 DataFrame 存储推荐评分
    ratings_df = pd.DataFrame(recommended_ratings, columns=['Anime Name', 'Predicted Rating'])
    
    # 选出评分最高的前 N 部动漫
    top_recommendations = ratings_df.sort_values(by='Predicted Rating', ascending=False).head(top_n)
    
    return top_recommendations
def fill_zero_ratings(similarity_matrix, matrix, top_n=5000):
    filled_matrix = matrix.copy()
    original_ratings = {}  # 用于存储原始评分

    # 仅选择前500个用户评分
    user_columns = filled_matrix.columns[:500]

    # 遍历每个动漫
    for anime_index in filled_matrix.index:
        for user_index in user_columns:
            if filled_matrix.loc[anime_index, user_index] > 0:  # 记录原始评分
                original_ratings[(anime_index, user_index)] = filled_matrix.loc[anime_index, user_index]
                filled_matrix.loc[anime_index, user_index] = 0  # 将评分变为0

    # 重新计算评分
    for anime_index in filled_matrix.index:
        for user_index in user_columns:
            if (anime_index, user_index) not in original_ratings:
                recommended_rating = item_based_recommendation(
                    similarity_matrix,
                    filled_matrix,
                    top_n,
                    user_index,
                    anime_index
                )
                filled_matrix.loc[anime_index, user_index] = recommended_rating

    return filled_matrix, original_ratings

# 计算均方误差
def calculate_mae(original_ratings, filled_matrix):
    y_true = []
    y_pred = []

    for (anime_index, user_index), original_rating in original_ratings.items():
        y_true.append(original_rating)
        y_pred.append(filled_matrix.loc[anime_index, user_index])

    return mean_absolute_error(y_true, y_pred)