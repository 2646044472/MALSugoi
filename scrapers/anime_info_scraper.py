import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException

# Configure Edge browser
def setup_browser():
    edge_options = Options()
    edge_options.add_argument("--disable-infobars")
    edge_options.add_argument("--disable-extensions")
    edge_options.add_argument("--log-level=3")
    edge_options.add_argument("--start-maximized")
    # edge_options.add_argument("--headless")  # Enable headless mode if needed

    service = Service(EdgeChromiumDriverManager().install())
    browser = webdriver.Edge(service=service, options=edge_options)
    return browser

# Main scraping function with recovery logic
def scrape_anime_rank_range(browser, start_rank, end_rank):
    anime_list = []
    batch_size = 100  # Save data every 200 items
    current_batch_start = start_rank  # Track the start of the current batch
    last_page_source = None  # Used to check for page loading issues

    # Loop through anime ranks in increments of 50 (as per MAL's pagination)
    for rank in range(start_rank, end_rank, 50):
        page = (rank // 50) * 50  # Calculate the page offset
        url = f"https://myanimelist.net/topanime.php?limit={page}"
        print(f"Accessing: {url}")

        retries = 3
        while retries > 0:
            try:
                browser.get(url)
                WebDriverWait(browser, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ranking-list"))
                )
                print("Page loaded successfully!")

                page_source = browser.page_source

                # Check if the page content has changed to avoid stale data
                if last_page_source == page_source:
                    print(f"No new content loaded, retrying...")
                    raise Exception("No new content")

                last_page_source = page_source  # Update last page source
                break  # Exit retry loop on success
            except WebDriverException:
                print("Browser may have been closed manually. Restarting browser...")
                browser = setup_browser()  # Restart the browser
                retries -= 1
                if retries == 0:
                    print(f"Retry limit reached, restarting browser again...")
                    browser.quit()
                    browser = setup_browser()
                    retries = 3  # Reset retry counter
                    last_page_source = None  # Reset page source
            except Exception as e:
                print(f"Page load failed: {e}")
                retries -= 1
                if retries == 0:
                    print(f"Retry limit reached, restarting browser again...")
                    browser.quit()
                    browser = setup_browser()
                    retries = 3  # Reset retry counter
                    last_page_source = None

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(last_page_source, 'html.parser')
        anime_entries = soup.find_all('tr', {'class': 'ranking-list'})

        # Loop through each anime entry on the page
        for i, entry in enumerate(anime_entries):
            current_rank = page + i  # Calculate the current rank
            if current_rank < start_rank or current_rank >= end_rank:
                continue  # Skip ranks outside the specified range

            title_tag = entry.find('h3', {'class': 'anime_ranking_h3'})
            title = title_tag.text.strip() if title_tag else 'N/A'

            score_tag = entry.find('td', {'class': 'score'})
            score = score_tag.text.strip() if score_tag else 'N/A'

            anime_url_tag = title_tag.find('a') if title_tag else None
            anime_url = anime_url_tag['href'] if anime_url_tag else 'N/A'

            print(f"Rank {current_rank} Info:")
            print(f"Title: {title}")
            print(f"Score: {score}")
            print(f"Details URL: {anime_url}")

            # Scrape detailed anime info if available
            anime_details = get_anime_details(browser, anime_url) if anime_url != 'N/A' else {
                'details_score': 'N/A',
                'ranked': 'N/A',
                'popularity': 'N/A',
                'members': 'N/A',
                'favorites': 'N/A',
                'genres': 'N/A'
            }

            # Append anime info to the list
            anime_list.append({
                'rank': current_rank,
                'title': title,
                'score': score,
                'url': anime_url,
                **anime_details  # Merge in the detailed info
            })

            # Save the data in batches
            if len(anime_list) >= batch_size:
                current_batch_end = current_batch_start + batch_size
                file_name = f"data/anime_info/anime_data_{current_batch_start}_to_{current_batch_end}.csv"
                save_data_to_csv(anime_list, file_name)
                anime_list.clear()  # Clear the list after saving
                current_batch_start = current_batch_end  # Update batch start

        time.sleep(1)  # Pause between page requests

    # Save any remaining data after the loop finishes
    if anime_list:
        current_batch_end = current_batch_start + len(anime_list)
        file_name = f"data/anime_info/anime_data_{current_batch_start}_to_{current_batch_end}.csv"
        save_data_to_csv(anime_list, file_name)

# Scrape detailed anime info from individual anime pages
def get_anime_details(browser, anime_url):
    anime_info = {}
    try:
        print(f"Accessing details page: {anime_url}")
        browser.get(anime_url)

        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "anime-detail-header-stats"))
        )

        page_source = browser.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        score_tag = soup.find('span', {'itemprop': 'ratingValue'})
        anime_info['details_score'] = score_tag.text.strip() if score_tag else 'N/A'

        ranked_tag = soup.find(string='Ranked:')
        anime_info['ranked'] = ranked_tag.parent.next_sibling.strip() if ranked_tag and ranked_tag.parent else 'N/A'

        popularity_tag = soup.find(string='Popularity:')
        anime_info['popularity'] = popularity_tag.parent.next_sibling.strip() if popularity_tag and popularity_tag.parent else 'N/A'

        members_tag = soup.find(string='Members:')
        anime_info['members'] = members_tag.parent.next_sibling.strip() if members_tag and members_tag.parent else 'N/A'

        favorites_tag = soup.find(string='Favorites:')
        anime_info['favorites'] = favorites_tag.parent.next_sibling.strip() if favorites_tag and favorites_tag.parent else 'N/A'

        genres_tag = soup.find_all('span', {'itemprop': 'genre'})
        genres = [genre.text.strip() for genre in genres_tag]
        anime_info['genres'] = ', '.join(genres) if genres else 'N/A'

    except Exception as e:
        print(f"Failed to load details page: {e}")
        anime_info = {
            'details_score': 'N/A',
            'ranked': 'N/A',
            'popularity': 'N/A',
            'members': 'N/A',
            'favorites': 'N/A',
            'genres': 'N/A'
        }

    return anime_info

# Save data to CSV
def save_data_to_csv(anime_list, file_name):
    df = pd.DataFrame(anime_list)

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"Data saved to {file_name}")

# Main function
def main(start_rank, end_rank):
    browser = setup_browser()
    while True:  # Infinite loop to allow multiple restarts
        try:
            scrape_anime_rank_range(browser, start_rank, end_rank)
            break  # Exit the loop if scraping completes successfully
        except WebDriverException:
            print("Browser was manually closed. Restarting...")
            browser = setup_browser()  # Restart the browser and continue
        except Exception as e:
            print(f"An error occurred: {e}. Restarting browser...")
            browser = setup_browser()  # Restart the browser in case of any other error
        finally:
            browser.quit()

# Execute the scraping process
if __name__ == "__main__":
    start_rank = 2600  # Starting rank for scraping
    end_rank = 6000  # Ending rank for scraping
    main(start_rank, end_rank)