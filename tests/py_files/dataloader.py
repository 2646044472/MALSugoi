import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 进度条库

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

# 准备训练数据
def construct_training_data(merged_data, anime_data, anime_features):
    """
    构建训练数据 (X, y)
    """
    X = []
    y = []
    
    grouped = merged_data.groupby('username')  # 按用户名分组
    
    # 使用 tqdm 包装分组数据，显示进度条
    for username, group in tqdm(grouped, desc="Processing Users", total=len(grouped)):
        user_ratings = group[['anime', 'rating']].set_index('anime')['rating'].to_dict()  # 用户的评分历史
        
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
    
    return np.array(X), np.array(y)

# 构建训练数据
X, y = construct_training_data(merged_data, anime_data, anime_features)

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