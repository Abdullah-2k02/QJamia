import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
    }

#paste the link/url of the page from which you want to fetch/scrap the data
url=""
response = requests.get(url,headers=headers)
html_content = response.content

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Get the text from the entire page without HTML tags
page_text = soup.get_text()

# Define the file name
file_name = 'programs offered.txt'

# Write the text content to a new file
with open(file_name, 'w', encoding='utf-8') as file:
    file.write(page_text)

print(f"Content saved to '{file_name}'")