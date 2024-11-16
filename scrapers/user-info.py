import requests
from bs4 import BeautifulSoup
import csv

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

# Base URL In this WEB we choose users whose age from 18 to 90
base_url = "https://myanimelist.net/users.php?cat=user&q=&loc=&agelow=18&agehigh=90&g="

# Number of pages to scrape
num_pages = 21

# Scrape the usernames from multiple pages
usernames = scrape_usernames(base_url, num_pages)

# Save the usernames to a CSV file
save_usernames_to_csv(usernames)

# Print the total count
print(f"\nTotal Usernames Extracted: {len(usernames)}")
