import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取CSV文件
df = pd.read_csv('anime_data.csv')

# 查看数据结构
print(df.head())

# 预处理: 将genres列中的空格去除并处理为小写
df['genres'] = df['genres'].apply(lambda x: x.replace(" ", "").lower())

# 使用TF-IDF向量化genres标签
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# 输出TF-IDF矩阵的形状
print(tfidf_matrix.shape)

# 计算余弦相似度矩阵
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 输出余弦相似度矩阵的形状
print(cosine_sim.shape)

# 构建一个从标题到索引的映射
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# 推荐函数
def get_recommendations(title, cosine_sim=cosine_sim):
    # 获取动画的索引
    idx = indices[title]
    
    # 获取所有动画的相似度分数，按降序排序
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 取出最相似的前10部动画（排除自身）
    sim_scores = sim_scores[1:11]
    
    # 获取相似动画的索引
    anime_indices = [i[0] for i in sim_scores]
    
    # 返回相似动画的标题
    return df['title'].iloc[anime_indices]

# 测试推荐系统
print(get_recommendations('Steins;Gate'))

# 重新计算推荐分数，加入评分权重
def get_recommendations_with_scores(title, cosine_sim=cosine_sim):
    # 获取动画的索引
    idx = indices[title]
    
    # 获取所有动画的相似度分数，按降序排序
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 加入评分权重，调整相似度（假设评分权重加0.1）
    sim_scores = [(i, score * df['score'].iloc[i]) for i, score in sim_scores]
    
    # 按照分数排序，取前10个
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    # 获取相似动画的索引
    anime_indices = [i[0] for i in sim_scores]
    
    # 返回相似动画的标题
    return df['title'].iloc[anime_indices]

# 测试推荐系统，带评分权重
print(get_recommendations_with_scores('Steins;Gate'))