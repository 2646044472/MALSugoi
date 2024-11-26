import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Function to get the HTML code of a URL
def fetch_html(url):
    """
    Fetches the HTML content of a given URL.
    
    Args:
        url (str): The target URL to fetch HTML from.
    
    Returns:
        str: HTML content of the URL or an error message.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Raise an error if the request fails
        response.raise_for_status()
        
        # Return the HTML content
        return response.text
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Function to extract usernames
def extract_usernames(html_code):
    """
    Extract all usernames from divs in the form:
    <div style="margin-bottom: 7px;"><a href="...">username</a></div>
    
    Args:
        html_code (str): The HTML content to parse.
    
    Returns:
        list: A list of extracted usernames.
    """
    # Parse the HTML
    soup = BeautifulSoup(html_code, 'html.parser')
    
    # Find all divs with the specific style
    matching_divs = soup.find_all('div', style="margin-bottom: 7px;")
    
    # Extract the username from each div
    usernames = [div.find('a').text for div in matching_divs]
    
    return usernames

# Function to loop through multiple pages and extract usernames
def scrape_usernames(base_url, num_pages):
    """
    Scrape usernames from multiple pages.
    
    Args:
        base_url (str): The base URL to scrape from.
        num_pages (int): The number of pages to scrape.
    
    Returns:
        list: A combined list of all extracted usernames.
    """
    all_usernames = []
    
    for i in range(num_pages):
        # Calculate the `show` parameter for pagination
        offset = i * 24
        
        # Construct the URL for the current page
        if i == 0:
            page_url = f"{base_url}"
        else:
            page_url = f"{base_url}&show={offset}"
        
        # Fetch the HTML code
        html_code = fetch_html(page_url)
        
        # Check if fetch was successful
        if not html_code.startswith("An error occurred"):
            # Extract usernames and add them to the list
            all_usernames.extend(extract_usernames(html_code))
        else:
            print(f"Failed to fetch page {i + 1}: {html_code}")
    
    return all_usernames

# Function to save usernames to a CSV file
def save_usernames_to_csv(usernames, filename="usernames.csv"):
    """
    Save the usernames to a CSV file.
    
    Args:
        usernames (list): List of usernames to save.
        filename (str): Name of the output CSV file.
    """
    if not usernames:
        print("No usernames to save.")
        return
    
    # Write the usernames to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Username"])  # Header row
        for username in usernames:
            writer.writerow([username])

    print(f"Usernames saved to {filename}")
def save_anime_info_to_csv(username, anime_info, filename="anime_info.csv"):
    if not anime_info:
        print(f"No anime info to save for {username}.")
        return
    
    # Write the anime info to a CSV file
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for anime_title, score in anime_info:
            writer.writerow([username, anime_title, score])

    print(f"Anime info saved for {username} to {filename}.")
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

    return anime_info

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

# Base URL In this WEB we choose users whose age from 18 to 90
base_url = "https://myanimelist.net/users.php?cat=user&q=&loc=&agelow=18&agehigh=90&g="

# Number of pages to scrape
num_pages = 42

# Scrape the usernames from multiple pages
usernames = scrape_usernames(base_url, num_pages)
for username in usernames:
    anime_url = f"https://myanimelist.net/animelist/{username}"
    html_code = fetch_dynamic_html(anime_url)
    if not html_code.startswith("An error occurred"):
        anime_info = extract_anime_info(html_code)
        save_anime_info_to_csv(username, anime_info)  # Save to CSV
    else:
        print(f"Failed to fetch anime list for {username}: {html_code}")