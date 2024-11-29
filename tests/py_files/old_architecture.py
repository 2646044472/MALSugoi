'''
### 1

class AnimeRecommendationModel(nn.Module):
    def __init__(self, num_users, num_anime, embedding_dim=50, num_genres):
        super(AnimeRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.anime_embedding = nn.Embedding(num_embeddings=num_anime, embedding_dim=embedding_dim)
        self.num_genres = num_genres
        self.fc1 = nn.Linear(embedding_dim * 2 + 3 + num_genres, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
    
    def forward(self, user_id, anime_id, anime_meta, genre_ids):
        user_embedded = self.user_embedding(user_id)
        anime_embedded = self.anime_embedding(anime_id)
        mask = (genre_ids != -1)
        genre_ids = genre_ids * mask
        genre_embedded = F.one_hot(genre_ids, num_classes=self.num_genres).float()
        genre_embedded = genre_embedded * mask.unsqueeze(-1)
        genre_embedded = torch.mean(genre_embedded, dim=1)
        x = torch.cat([user_embedded, anime_embedded, anime_meta, genre_embedded], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

model = AnimeRecommendationModel(num_users=num_users, num_anime=num_anime, embedding_dim=50, num_genres=num_genres).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### 2
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimeRecommendationModel(nn.Module):
    def __init__(self, num_users, num_animes, num_genres, embed_dim=32, dropout_rate=0.3):
        super(AnimeRecommendationModel, self).__init__()
        
        # Embedding layers for user and anime
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.anime_embedding = nn.Embedding(num_embeddings=num_animes, embedding_dim=embed_dim)
        
        # Linear layer for anime meta features (score, members, favorites)
        self.anime_meta_fc = nn.Linear(3, 16)  # 将 3 个连续特征映射到 16 维
        
        # Fully connected layers
        self.fc1 = nn.Linear(embed_dim * 2 + 16 + num_genres, 256)  # 输入维度为嵌入 + meta + genre
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)  # 输出评分
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, user_id, anime_id, anime_meta, genre_ids):
        # User embedding
        user_embedded = self.user_embedding(user_id)  # (batch_size, embed_dim)
        
        # Anime embedding
        anime_embedded = self.anime_embedding(anime_id)  # (batch_size, embed_dim)
        
        # Anime meta features (score, members, favorites)
        anime_meta_processed = self.anime_meta_fc(anime_meta)  # (batch_size, 16)
        anime_meta_processed = F.relu(anime_meta_processed)
        
        # Genre processing: one-hot encode and average pooling
        mask = (genre_ids != -1)  # Mask to handle padding or invalid genre IDs
        genre_ids = genre_ids * mask
        genre_embedded = F.one_hot(genre_ids, num_classes=num_genres).float()  # 独热编码，(batch_size, max_genre_count, num_genres)
        genre_embedded = genre_embedded * mask.unsqueeze(-1)  # Apply mask
        genre_embedded = torch.mean(genre_embedded, dim=1)  # Average pooling, (batch_size, num_genres)
        
        # Concatenate all features
        x = torch.cat([user_embedded, anime_embedded, anime_meta_processed, genre_embedded], dim=1)
        
        # Fully connected layers with Dropout and BatchNorm
        x = self.dropout(F.relu(self.fc1(x)))  # First FC layer
        x = self.bn1(x)  # Batch normalization
        x = self.dropout(F.relu(self.fc2(x)))  # Second FC layer
        x = self.bn2(x)  # Batch normalization
        x = F.relu(self.fc3(x))  # Third FC layer
        
        # Output layer
        x = self.output(x)
        return x
    
### 3
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimeRecommendationModel(nn.Module):
    def __init__(self, num_users, num_animes, num_genres, embed_dim=32):
        super(AnimeRecommendationModel, self).__init__()
        
        # Embedding layers for user and anime
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.anime_embedding = nn.Embedding(num_embeddings=num_animes, embedding_dim=embed_dim)
        
        # Linear layer for anime meta features (score, members, favorites)
        self.anime_meta_fc = nn.Linear(3, 8)  # 将 3 个连续特征映射到 8 维

        # Fully connected layers
        self.fc = nn.Linear(embed_dim * 2 + 8 + num_genres, 1)  # 直接输出预测评分

    def forward(self, user_id, anime_id, anime_meta, genre_ids):
        # User embedding
        user_embedded = self.user_embedding(user_id)  # (batch_size, embed_dim)
        
        # Anime embedding
        anime_embedded = self.anime_embedding(anime_id)  # (batch_size, embed_dim)
        
        # Anime meta features (score, members, favorites)
        anime_meta_processed = self.anime_meta_fc(anime_meta)  # (batch_size, 8)
        anime_meta_processed = F.relu(anime_meta_processed)

        # Genre processing: one-hot encode and average pooling
        mask = (genre_ids != -1)  # Mask to handle padding or invalid genre IDs
        genre_ids = genre_ids * mask
        genre_embedded = F.one_hot(genre_ids, num_classes=num_genres).float()  # (batch_size, max_genre_count, num_genres)
        genre_embedded = genre_embedded * mask.unsqueeze(-1)
        genre_embedded = torch.mean(genre_embedded, dim=1)  # Average pooling, (batch_size, num_genres)
        
        # Concatenate all features
        x = torch.cat([user_embedded, anime_embedded, anime_meta_processed, genre_embedded], dim=1)
        
        # Fully connected layer (directly output the prediction)
        x = self.fc(x)  # (batch_size, 1)
        return x.squeeze(1)  # 输出形状为 (batch_size,)
    

### 4
import torch
import torch.nn as nn


class SimplestAnimeRecommendationModel(nn.Module):
    def __init__(self, num_users, num_animes, embed_dim=32):
        super(SimplestAnimeRecommendationModel, self).__init__()
        
        # Embedding layers for user and anime
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.anime_embedding = nn.Embedding(num_embeddings=num_animes, embedding_dim=embed_dim)
        
        # Fully connected layer
        self.fc = nn.Linear(embed_dim * 2, 1)  # 直接输出预测评分

    def forward(self, user_id, anime_id):
        # User embedding
        user_embedded = self.user_embedding(user_id)  # (batch_size, embed_dim)
        
        # Anime embedding
        anime_embedded = self.anime_embedding(anime_id)  # (batch_size, embed_dim)
        
        # Concatenate user and anime embeddings
        x = torch.cat([user_embedded, anime_embedded], dim=1)  # (batch_size, embed_dim * 2)
        
        # Fully connected layer (directly output the prediction)
        x = self.fc(x)  # (batch_size, 1)
        return x.squeeze(1)  # 输出形状为 (batch_size,)

'''