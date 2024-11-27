import pandas as pd
import numpy as np

def cosine(a, b):
    # 计算余弦相似度
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def df_to_matrix(df):
    data = []
    users = set()
    animes = set()

    for index, row in df.iterrows():
        user, anime, rating = row['username'], row['anime'], row['rating']
        # 将评分转换为浮点数，处理缺失值
        if rating == '-' or pd.isna(rating):
            rating = np.nan
        else:
            rating = float(rating)  # 确保将评分转换为浮点数
        
        data.append((user, anime, rating))
        users.add(user)
        animes.add(anime)

    users = sorted(list(users))
    animes = sorted(list(animes))

    matrix = pd.DataFrame(np.nan, index=animes, columns=users)

    for user, anime, rating in data:
        matrix.at[anime, user] = rating

    matrix = matrix.fillna(0)  # 将 NaN 替换为 0
    return matrix

def calculate_similarity_matrix(rating_matrix):
    num_animes = rating_matrix.shape[0]
    similarity_matrix = pd.DataFrame(np.zeros((num_animes, num_animes)), 
                                      columns=rating_matrix.index, 
                                      index=rating_matrix.index)

    for i in range(num_animes):
        for j in range(i, num_animes):
            # 计算余弦相似度
            sim = cosine(rating_matrix.iloc[i, :].values, rating_matrix.iloc[j, :].values)
            similarity_matrix.iloc[i, j] = sim
            similarity_matrix.iloc[j, i] = sim  # 对称赋值

    return similarity_matrix

# Example usage
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/anime_info.csv'
df = pd.read_csv(csv_file_path)
rating_matrix = df_to_matrix(df)
