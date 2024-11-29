import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# 1. 加载数据
anime_data = pd.read_csv('./data/anime_info/anime_data.csv')
user_ratings = pd.read_csv('./data/user_animelist/anime_info.csv')

# 清理数据
anime_data['score'] = anime_data['score'].replace('-', np.nan).astype(float)  # 如果有缺失值处理
user_ratings['rating'] = user_ratings['rating'].replace('-', np.nan).astype(float)

# 去掉没有评分的条目
user_ratings = user_ratings.dropna(subset=['rating'])

# 格式化数值
anime_data['members'] = anime_data['members'].str.replace(',', '').astype(float)
anime_data['favorites'] = anime_data['favorites'].str.replace(',', '').astype(float)
anime_data['popularity'] = anime_data['popularity'].str.replace('#', '').astype(float)
anime_data['ranked'] = anime_data['ranked'].str.replace('#', '').astype(float)

# 2. 提取番剧特征
def preprocess_genres(genres_series):
    """将 genres 列转为 multi-hot 编码"""
    genres_series = genres_series.fillna('')  # 填充空值
    genres_list = genres_series.str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(genres_list)
    return genres_encoded, mlb.classes_

# 处理 genres 列
genres_encoded, genres_classes = preprocess_genres(anime_data['genres'])

# 将 genres 编码加入 anime_data
anime_data = anime_data.join(pd.DataFrame(genres_encoded, columns=genres_classes))

# 特征列
anime_features = ['score', 'ranked', 'popularity', 'members', 'favorites'] + list(genres_classes)

# 标准化数值特征
scaler = StandardScaler()
anime_data[anime_features] = scaler.fit_transform(anime_data[anime_features])

# 3. 构建用户-番剧评分的训练数据
# 按照 anime title 合并 user_ratings 和 anime_data
merged_data = user_ratings.merge(anime_data, left_on='anime', right_on='title', how='inner')

# 单个用户数据处理函数
def process_user_data(username, group, anime_data, anime_features):
    """
    处理单个用户的数据，返回 (X, y)
    """
    user_ratings = group[['anime', 'rating']].set_index('anime')['rating'].to_dict()  # 用户的评分历史
    
    X = []
    y = []
    
    for _, row in group.iterrows():
        # 当前目标番剧
        target_anime = row['anime']
        target_rating = row['rating']
        
        # 跳过如果没有评分的目标番剧
        if target_anime not in user_ratings:
            continue
        
        # 构建输入特征
        user_history = {k: v for k, v in user_ratings.items() if k != target_anime}  # 除去目标番剧的历史
        history_features = []
        
        for anime, rating in user_history.items():
            if anime in anime_data['title'].values:
                anime_row = anime_data[anime_data['title'] == anime][anime_features].iloc[0].values
                history_features.append(anime_row * rating)  # 特征加权
        
        # 如果用户历史为空，则跳过
        if len(history_features) == 0:
            continue
        
        # 聚合历史特征（例如求平均值）
        history_features = np.mean(history_features, axis=0)
        
        # 目标番剧的特征
        target_features = anime_data[anime_data['title'] == target_anime][anime_features].iloc[0].values
        
        # 拼接特征
        input_features = np.concatenate([history_features, target_features])
        
        # 添加到训练集
        X.append(input_features)
        y.append(target_rating)
    
    return X, y

# 使用 Joblib 并行处理用户数据
def construct_training_data_joblib(merged_data, anime_data, anime_features, n_jobs=10):
    """
    使用 joblib 并行构建训练数据
    """
    grouped = merged_data.groupby('username')  # 按用户名分组
    user_groups = [(username, group) for username, group in grouped]

    # 并行处理
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_user_data)(username, group, anime_data, anime_features)
        for username, group in tqdm(user_groups, desc="Processing Users")
    )

    # 合并所有结果
    X = []
    y = []
    for user_X, user_y in results:
        X.extend(user_X)
        y.extend(user_y)
    
    return np.array(X), np.array(y)

# 构建训练数据（Joblib 版）
X, y = construct_training_data_joblib(merged_data, anime_data, anime_features, n_jobs=10)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 数据集
class AnimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AnimeDataset(X_train, y_train)
test_dataset = AnimeDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 输出数据形状
print(f"训练集 X 形状: {X_train.shape}, y 形状: {y_train.shape}")
print(f"测试集 X 形状: {X_test.shape}, y 形状: {y_test.shape}")

# 4. 保存和加载数据
def save_data(X, y, X_train, X_test, y_train, y_test, save_path='./processed_data/'):
    """
    保存训练数据和测试数据到文件
    """
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录
    
    # 保存 NumPy 数组
    np.save(save_path + 'X.npy', X)
    np.save(save_path + 'y.npy', y)
    np.save(save_path + 'X_train.npy', X_train)
    np.save(save_path + 'X_test.npy', X_test)
    np.save(save_path + 'y_train.npy', y_train)
    np.save(save_path + 'y_test.npy', y_test)
    
    print(f"数据保存到 {save_path}")

def load_data(save_path='./processed_data/'):
    """
    从文件中加载训练数据和测试数据
    """
    X = np.load(save_path + 'X.npy')
    y = np.load(save_path + 'y.npy')
    X_train = np.load(save_path + 'X_train.npy')
    X_test = np.load(save_path + 'X_test.npy')
    y_train = np.load(save_path + 'y_train.npy')
    y_test = np.load(save_path + 'y_test.npy')
    
    print(f"数据从 {save_path} 加载完成")
    return X, y, X_train, X_test, y_train, y_test

# 保存数据
save_data(X, y, X_train, X_test, y_train, y_test)

# 示例：加载数据
# X, y, X_train, X_test, y_train, y_test = load_data()