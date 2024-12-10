import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# Define the AnimeRatingPredictor model


class AnimeRatingPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout=0.3):
        """
        Fully connected neural network for anime rating prediction.
        :param input_size: Input feature dimension
        :param hidden_sizes: List of hidden layer sizes
        :param dropout: Dropout probability
        """
        super(AnimeRatingPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])  # Batch Normalization
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass
        :param x: Input features
        :return: Predicted rating in the range [1, 10]
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output(x)
        x = torch.sigmoid(x) * 9 + 1  # Map to [1, 10]
        return x


# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "recommenders_DNN/model/model.pth"
model = torch.load(model_path, map_location=device)
model.eval()  # Set model to evaluation mode
print("Model loaded successfully.")

# Load and preprocess anime feature data
anime_data = pd.read_csv('./data/anime_info/anime_data.csv')

# Data cleaning
anime_data['score'] = anime_data['score'].replace('-', 0).astype(float)
anime_data['members'] = anime_data['members'].str.replace(
    ',', '').astype(float)
anime_data['favorites'] = anime_data['favorites'].str.replace(
    ',', '').astype(float)
anime_data['popularity'] = anime_data['popularity'].str.replace(
    '#', '').astype(float)
anime_data['ranked'] = anime_data['ranked'].str.replace(
    '#', '').replace('-', 0).astype(float)

# Encode genres


def preprocess_genres(genres_series):
    genres_series = genres_series.fillna('')
    genres_list = genres_series.str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(genres_list)
    return genres_encoded, mlb.classes_


genres_encoded, genres_classes = preprocess_genres(anime_data['genres'])
anime_data = anime_data.join(pd.DataFrame(
    genres_encoded, columns=genres_classes))

# Define feature columns
anime_features = ['score', 'ranked', 'popularity',
                  'members', 'favorites'] + list(genres_classes)

# Standardize features
scaler = StandardScaler()
anime_data[anime_features] = scaler.fit_transform(anime_data[anime_features])

# Convert features to tensors
anime_tensor = torch.tensor(
    anime_data[anime_features].values, dtype=torch.float32).to(device)
titles = anime_data['title'].values
title_to_index = {title: idx for idx, title in enumerate(titles)}

# Load user rating data
user_input = pd.read_csv('./recommenders_DNN/test/input.txt',
                         header=None, names=['username', 'anime', 'rating'])
user_input['rating'] = user_input['rating'].replace(
    '-', 0).astype(float)  # Convert '-' to 0

# Extract user history
user_history = user_input[user_input['rating'] > 0][[
    'anime', 'rating']].set_index('anime')['rating'].to_dict()

# Recommendation function


def generate_recommendations(model, user_history, anime_tensor, title_to_index, top_k):
    """
    Generate recommendations based on user history.
    """
    history_features = []
    for anime, rating in user_history.items():
        if anime in title_to_index:
            anime_idx = title_to_index[anime]
            anime_feature = anime_tensor[anime_idx]
            history_features.append(anime_feature * rating)

    if len(history_features) == 0:
        return []

    # Aggregate history features (mean pooling)
    history_features = torch.stack(history_features).mean(dim=0)

    # Predict ratings for all unseen anime
    predictions = {}
    for title, idx in title_to_index.items():
        if title not in user_history:  # Recommend only unseen anime
            input_features = torch.cat([history_features, anime_tensor[idx]])
            input_features = input_features.unsqueeze(0)  # Add batch dimension
            predicted_rating = model(input_features.to(device)).item()
            predictions[title] = predicted_rating

    # Sort by predicted ratings and return top K
    recommended_titles = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return recommended_titles


# Generate recommendations
recommendations = generate_recommendations(
    model, user_history, anime_tensor, title_to_index, top_k=50)

# Save recommendations to a DataFrame
recommendations_df = pd.DataFrame(recommendations, columns=[
                                  "Title", "Predicted Rating"])

# Print the top 50 recommendations
print(recommendations_df.head(50))
