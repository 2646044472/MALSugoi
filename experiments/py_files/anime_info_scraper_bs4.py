import aiohttp
import asyncio
from bs4 import BeautifulSoup
from tqdm import tqdm  # Progress bar library
import pandas as pd
import os

# Base URL and headers for requests
base_url = 'https://myanimelist.net/topanime.php'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# Limit concurrent tasks
concurrent_limit = 10
semaphore = asyncio.Semaphore(concurrent_limit)

# Asynchronously fetch page content
async def fetch(session, url):
    async with semaphore:
        async with session.get(url) as response:
            return await response.text()

# Asynchronously scrape anime details
async def scrape_anime_details(session, anime_url, title, score, anime_list):
    anime_info = {}
    try:
        anime_page = await fetch(session, anime_url)
        anime_soup = BeautifulSoup(anime_page, 'html.parser')

        # Extract genres
        genres = [genre.text.strip() for genre in anime_soup.find_all('span', {'itemprop': 'genre'})]
        anime_info['Genres'] = ', '.join(genres)

        # Extract other details like score, rank, popularity, members, and favorites
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

        # Append the extracted information to the list
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

# Asynchronously scrape the top anime list
async def scrape_top_anime(session, start_anime, end_anime):
    anime_list = []
    page = start_anime // 50  # Calculate the starting page
    scraped_animes = 0
    total_animes = end_anime - start_anime

    # Progress bar
    with tqdm(total=total_animes, desc="Scraping Anime") as pbar:
        while scraped_animes < total_animes:
            # Construct page URL
            url = f"{base_url}?limit={page * 50}"

            # Fetch page content
            page_content = await fetch(session, url)
            soup = BeautifulSoup(page_content, 'html.parser')

            # Find all anime entries on the page
            anime_entries = soup.find_all('tr', {'class': 'ranking-list'})

            tasks = []
            for i, entry in enumerate(anime_entries):
                current_index = page * 50 + i
                if current_index < start_anime:
                    continue
                if current_index >= end_anime:
                    break

                # Extract title and score
                title_tag = entry.find('h3', {'class': 'anime_ranking_h3'})
                title = title_tag.text.strip() if title_tag else 'Title not found'

                score_tag = entry.find('td', {'class': 'score'})
                score = score_tag.text.strip() if score_tag else 'Score not found'

                # Extract anime details URL
                anime_url_tag = title_tag.find('a') if title_tag else None
                anime_url = anime_url_tag['href'] if anime_url_tag else None

                # Asynchronously scrape anime details
                if anime_url:
                    tasks.append(scrape_anime_details(session, anime_url, title, score, anime_list))

                scraped_animes += 1
                if scraped_animes >= total_animes:
                    break

            # Wait for all tasks to finish
            await asyncio.gather(*tasks)

            # Update progress bar
            pbar.update(len(tasks))

            # Move to the next page
            page += 1

    return anime_list

# Main function
async def main(start_anime, end_anime):
    async with aiohttp.ClientSession(headers=headers) as session:
        anime_list = await scrape_top_anime(session, start_anime, end_anime)
    return anime_list

# Save data to CSV file
def save_data_to_csv(anime_list, file_name):
    df = pd.DataFrame(anime_list)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"Data saved to {file_name}")

# Entry point
if __name__ == "__main__":
    # Set the range of anime to scrape
    start_anime = 501
    end_anime = 1000

    # Run the main function
    anime_list = asyncio.run(main(start_anime, end_anime))

    # Save the scraped data to a CSV file
    file_name = f"data/anime_info/anime_data_{start_anime}_to_{end_anime}.csv"
    save_data_to_csv(anime_list, file_name)