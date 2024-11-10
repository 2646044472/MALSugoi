from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager  # 自动管理Edge WebDriver
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# 配置 Selenium WebDriver for Edge
def setup_browser():
    edge_options = Options()
    # edge_options.add_argument("--start-maximized")  # 启动时最大化窗口
    edge_options.add_argument("--disable-infobars")  # 禁用浏览器正在被自动化工具控制的提示
    edge_options.add_argument("--disable-extensions")  # 禁用扩展

    # 使用 webdriver-manager 自动下载和管理 Edge WebDriver
    service = Service(EdgeChromiumDriverManager().install())
    
    browser = webdriver.Edge(service=service, options=edge_options)
    return browser

# 同步获取页面内容
def fetch(browser, url):
    print(f"正在获取页面: {url}")
    browser.get(url)
    try:
        # 等待页面加载完成，等待某个关键元素出现
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ranking-list"))
        )
        return browser.page_source
    except Exception as e:
        print(f"获取页面 {url} 时出错: {e}")
        return None

# 爬取单个动漫的详细信息
def scrape_anime_details(browser, anime_url, title, score, anime_list):
    anime_info = {}
    try:
        print(f"正在爬取动漫详情: {title} ({anime_url})")
        anime_page = fetch(browser, anime_url)
        if not anime_page:
            print(f"获取 {title} 的详情失败")
            return

        anime_soup = BeautifulSoup(anime_page, 'html.parser')

        # 提取标签和评分等信息
        genres = [genre.text.strip() for genre in anime_soup.find_all('span', {'itemprop': 'genre'})]
        anime_info['Genres'] = ', '.join(genres)

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

        # 保存动漫的信息
        anime_list.append({
            'title': title,
            'score': score,
            'genres': anime_info.get('Genres', 'N/A'),
            'ranked': anime_info.get('Ranked', 'N/A'),
            'popularity': anime_info.get('Popularity', 'N/A'),
            'members': anime_info.get('Members', 'N/A'),
            'favorites': anime_info.get('Favorites', 'N/A')
        })
        print(f"完成爬取: {title}")
    except Exception as e:
        print(f"爬取 {anime_url} 时出错: {e}")

# 爬取Top Anime列表
def scrape_top_anime(browser, start_anime, end_anime):
    anime_list = []
    page = start_anime // 50  # 计算从哪一页开始
    scraped_animes = 0
    total_animes = end_anime - start_anime

    while scraped_animes < total_animes:
        # 构建页面URL
        url = f"https://myanimelist.net/topanime.php?limit={page * 50}"
        print(f"正在获取第 {page} 页 ({url})")

        # 获取页面内容
        page_content = fetch(browser, url)
        if not page_content:
            print(f"获取第 {page} 页失败，跳过...")
            page += 1
            continue

        soup = BeautifulSoup(page_content, 'html.parser')

        # 找到所有动漫条目
        anime_entries = soup.find_all('tr', {'class': 'ranking-list'})
        print(f"在第 {page} 页找到 {len(anime_entries)} 条动漫")

        for i, entry in enumerate(anime_entries):
            current_index = page * 50 + i
            if current_index < start_anime:
                continue
            if current_index >= end_anime:
                break

            # 提取动漫的标题和评分
            title_tag = entry.find('h3', {'class': 'anime_ranking_h3'})
            title = title_tag.text.strip() if title_tag else 'Title not found'

            score_tag = entry.find('td', {'class': 'score'})
            score = score_tag.text.strip() if score_tag else 'Score not found'

            # 获取动漫详情页的URL
            anime_url_tag = title_tag.find('a') if title_tag else None
            anime_url = anime_url_tag['href'] if anime_url_tag else None

            # 爬取动漫的详细信息
            if anime_url:
                scrape_anime_details(browser, anime_url, title, score, anime_list)

            scraped_animes += 1
            if scraped_animes >= total_animes:
                break

        # 移动到下一页
        page += 1

        # 可选：在每一页之间添加延迟，避免被封
        # time.sleep(1)

    return anime_list

# 保存数据到CSV文件
def save_data_to_csv(anime_list, file_name):
    df = pd.DataFrame(anime_list)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {file_name}")

# 主函数
if __name__ == "__main__":
    # 设置要爬取的动漫范围
    start_anime = 500
    end_anime = 1000

    # 设置浏览器
    browser = setup_browser()

    # 爬取动漫列表
    anime_list = scrape_top_anime(browser, start_anime, end_anime)

    # 保存爬取结果到CSV文件
    file_name = f"data/anime_info/anime_data_{start_anime}_to_{end_anime}.csv"
    save_data_to_csv(anime_list, file_name)

    # 关闭浏览器
    browser.quit()