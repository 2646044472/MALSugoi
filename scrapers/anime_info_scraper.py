import aiohttp
import asyncio
from bs4 import BeautifulSoup
from tqdm import tqdm  # Progress bar library
import pandas as pd
import os

# 定义基本的URL和请求头
base_url = 'https://myanimelist.net/topanime.php'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# 存储抓取的动漫信息
anime_list = []

# 要抓取的最大数量
max_animes = 10000  

# 设置并发限制
concurrent_limit = 10
semaphore = asyncio.Semaphore(concurrent_limit)

# 异步获取页面内容
async def fetch(session, url):
    async with semaphore:  # 通过信号量限制并发任务
        async with session.get(url) as response:
            return await response.text()

# 异步抓取动漫详情
async def scrape_anime_details(session, anime_url, title, score):
    anime_info = {}
    try:
        anime_page = await fetch(session, anime_url)
        anime_soup = BeautifulSoup(anime_page, 'html.parser')

        # 1. 抓取“类型”部分
        genres = [genre.text.strip() for genre in anime_soup.find_all('span', {'itemprop': 'genre'})]
        anime_info['Genres'] = ', '.join(genres)

        # 2. 抓取“评分”、“排名”、“人气”、“成员”和“收藏”
        score_tag = anime_soup.find('span', {'itemprop': 'ratingValue'})
        if score_tag:
            anime_info['Score'] = score_tag.text.strip()

        ranked_tag = anime_soup.find(string='Ranked:')
        if ranked_tag and ranked_tag.parent:
            anime_info['Ranked'] = ranked_tag.parent.next_sibling.strip()

        popularity_tag = anime_soup.find(string='Popularity:')
        if popularity_tag and popularity_tag.parent:
            anime_info['Popularity'] = popularity_tag.parent.next_sibling.strip()

        members_tag = anime_soup.find(string='Members:')
        if members_tag and members_tag.parent:
            anime_info['Members'] = members_tag.parent.next_sibling.strip()

        favorites_tag = anime_soup.find(string='Favorites:')
        if favorites_tag and favorites_tag.parent:
            anime_info['Favorites'] = favorites_tag.parent.next_sibling.strip()

        # 将动漫信息添加到列表中
        anime_list.append({
            'title': title,
            'score': score,
            'genres': anime_info.get('Genres', 'N/A'),
            'ranked': anime_info.get('Ranked', 'N/A'),
            'popularity': anime_info.get('Popularity', 'N/A'),
            'members': anime_info.get('Members', 'N/A'),
            'favorites': anime_info.get('Favorites', 'N/A')
        })
    except Exception as e:
        print(f"Error scraping {anime_url}: {e}")

# 异步抓取Top动漫列表
async def scrape_top_anime(session):
    page = 0
    scraped_animes = 0

    with tqdm(total=max_animes, desc="Scraping Anime") as pbar:
        while scraped_animes < max_animes:
            # 构造分页的URL
            url = f"{base_url}?limit={page * 50}"

            # 获取页面内容
            page_content = await fetch(session, url)
            soup = BeautifulSoup(page_content, 'html.parser')

            # 找到页面中的所有动漫条目
            anime_entries = soup.find_all('tr', {'class': 'ranking-list'})

            tasks = []
            for entry in anime_entries:
                # 提取动漫的标题和评分
                title_tag = entry.find('h3', {'class': 'anime_ranking_h3'})
                title = title_tag.text.strip() if title_tag else 'Title not found'

                score_tag = entry.find('td', {'class': 'score'})
                score = score_tag.text.strip() if score_tag else 'Score not found'

                # 提取动漫详情页面的URL
                anime_url_tag = title_tag.find('a') if title_tag else None
                anime_url = anime_url_tag['href'] if anime_url_tag else None

                # 异步抓取动漫详情
                if anime_url:
                    tasks.append(scrape_anime_details(session, anime_url, title, score))

                # 如果达到抓取的最大数量则停止
                scraped_animes += 1
                if scraped_animes >= max_animes:
                    break

            # 等待所有任务完成
            await asyncio.gather(*tasks)

            # 更新进度条
            pbar.update(len(tasks))

            # 处理下一页
            page += 1

# 主函数
async def main():
    async with aiohttp.ClientSession(headers=headers) as session:
        await scrape_top_anime(session)

# 保存数据到文件
def save_data_to_csv(anime_list, file_path):
    # 将抓取数据转换为 DataFrame
    df = pd.DataFrame(anime_list)
    
    # 确保路径中的目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存到 CSV 文件
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Data saved to {file_path}")

# 启动程序
if __name__ == "__main__":
    # 使用 asyncio.run 启动 main 函数
    asyncio.run(main())

    # 数据保存路径
    data_path = "data/anime_data.csv"

    # 保存抓取到的数据到 CSV
    save_data_to_csv(anime_list, data_path)