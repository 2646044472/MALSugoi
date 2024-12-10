from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Function to fetch dynamic page source using Selenium
def fetch_dynamic_html(url):
    """
    Fetches the HTML content of a dynamic webpage using Selenium and Chrome.
    
    Args:
        url (str): The target URL to fetch.
    
    Returns:
        str: HTML content of the page.
    """
    try:
        # Set up Chrome options
        options = Options()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")  # Disable GPU rendering
        options.add_argument("--no-sandbox")  # Avoid sandbox issues

        # Set up the Chrome WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Load the webpage
        driver.get(url)
        
        # Wait for the page to load (adjust the time if needed)
        time.sleep(5)
        
        # Get the page source
        html = driver.page_source
        
        # Close the browser
        driver.quit()
        
        return html
    except Exception as e:
        return f"An error occurred: {e}"

# Function to extract anime info
def extract_anime_info(html_code):
    """
    Extracts the anime title and score from the user's anime list page HTML.

    Args:
        html_code (str): The HTML content of the user's anime list page.

    Returns:
        list: A list of tuples where each tuple contains (anime_title, score).
    """
    soup = BeautifulSoup(html_code, 'html.parser')
    anime_info = []

    # Find the username
    username = soup.find('h1', class_='username').text.strip() if soup.find('h1', class_='username') else "Unknown User"

    # Find all rows in the anime list table
    rows = soup.find_all('tr', class_='list-table-data')

    for row in rows:
        # Locate the container with title information
        title_container = row.find('td', class_='data title clearfix')
        if title_container:
            # Locate the anime title within the container
            title_tag = title_container.find('a', class_='link sort')
            anime_title = title_tag.text.strip() if title_tag else "Unknown"
        else:
            anime_title = "Unknown"

        # Extract the score
        score_container = row.find('td', class_='data score')
        score = (
            score_container.find('span', class_='score-label').text.strip()
            if score_container and score_container.find('span', class_='score-label')
            else "-"
        )

        # Append the extracted info as a tuple (anime_title, score)
        anime_info.append((anime_title, score))

    return username, anime_info

# Convert anime info to table and display
def display_anime_table(username, anime_info):
    """
    Converts anime information into a table format and displays it.

    Args:
        username (str): The username of the list owner.
        anime_info (list): List of tuples containing anime titles and scores.
    """
    # Convert data to a DataFrame
    df = pd.DataFrame(anime_info, columns=['Anime Title', 'Score'])
    
    # Print username and the table
    print(f"\nAnime List for User: {username}\n")
    print(df)

# Example usage
url = "https://myanimelist.net/animelist/cindia"
html_code = fetch_dynamic_html(url)
username, anime_info = extract_anime_info(html_code)
display_anime_table(username, anime_info)
