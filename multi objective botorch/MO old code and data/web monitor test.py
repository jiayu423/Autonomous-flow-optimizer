import requests
from bs4 import BeautifulSoup

url = "http://augurex-dibox-3/"
# url = "https://requests.readthedocs.io/en/latest/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')

print(soup.prettify())