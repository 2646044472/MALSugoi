### 最终输入特征
用户历史特征聚合 = [0.3, -0.5, 0.7]（假设历史特征聚合维度为 3）。
目标番剧特征 = [0.1, 0.2, -0.1]（假设目标番剧特征维度为 3）。
最终的输入特征是：[0.3, -0.5, 0.7, 0.1, 0.2, -0.1]

### 用户历史评分特征
anime_features = ['score', 'ranked', 'popularity', 'members', 'favorites'] + list(genres_classes)
对于 user1 的历史评分 {'One Piece': 9}：
One Piece 的特征 [0.2, 0.1, -0.1, ...] 被加权为：
[9 * 0.2, 9 * 0.1, 9 * -0.1, ...] = [1.8, 0.9, -0.9, ...]

### 目标番剧评分
历史评分里的相关番剧应被剔除以防数据泄露
anime_features = ['score', 'ranked', 'popularity', 'members', 'favorites'] + list(genres_classes)


