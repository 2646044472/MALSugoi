import pandas as pd
import numpy as np

def convert_to_int(rating_str):
    try:
        # 尝试将字符串转换为整数
        return int(rating_str)
    except ValueError:
        # 如果转换失败，返回0或其他默认值
        print(f"Warning: Cannot convert '{rating_str}' to an integer. Setting to 0.")
        return 0

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
            rating = '0'  # 将字符串 '0' 传递给 convert_to_int
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

def calculate_similarity_matrix(rating_matrix):
    num_animes = rating_matrix.shape[0]  # number of rows, shape[1] means number of columns
    similarity_matrix = pd.DataFrame(np.zeros((num_animes, num_animes)), 
                                      columns=rating_matrix.index, 
                                      index=rating_matrix.index)

    for i in range(num_animes):
        for j in range(i, num_animes):
            sim = cosine(rating_matrix.iloc[i, :].values, rating_matrix.iloc[j, :].values)
            similarity_matrix.iloc[i, j] = sim
            similarity_matrix.iloc[j, i] = sim

    return similarity_matrix

def calculate_item_similarity(rating_matrix):
    # Get the number of items (animes)
    num_items = rating_matrix.shape[0]
    
    # Initialize an empty DataFrame for the similarity matrix
    similarity_matrix = pd.DataFrame(np.zeros((num_items, num_items)), 
                                      columns=rating_matrix.index, 
                                      index=rating_matrix.index)

    # Loop through each pair of items (animes) to calculate similarity
    for i in range(num_items):
        for j in range(i, num_items):
            # Calculate cosine similarity between item i and item j
            sim = 1 - cosine(rating_matrix.iloc[i, :].values, rating_matrix.iloc[j, :].values)
            
            # Store the similarity in both the (i, j) and (j, i) positions (symmetric matrix)
            similarity_matrix.iloc[i, j] = sim
            similarity_matrix.iloc[j, i] = sim

    return similarity_matrix

# 示例用法
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/data/user_animelist/anime_info.csv'
df = pd.read_csv(csv_file_path)
matrix = df_to_matrix(df)
similarity_matrix = calculate_item_similarity(matrix)
print(similarity_matrix)
