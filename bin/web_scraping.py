import requests
from bs4 import BeautifulSoup

data = {}

def get_html(url):
    response = requests.get(url)
    return response.text


def scrap_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    numbers = soup.find_all('div', class_='P6K39c')
    text = soup.find_all('div', class_='mfs7Fc')
    for number, i in zip(numbers, text):
        print(i.text, number.text)
        data[i.text] = number.text
    print(data)

def main():
    key = 'NVDA'
    url = f'https://www.google.com/finance/quote/{key}:NASDAQ'
    html = get_html(url)
    scrap_html(html)


if __name__ == "__main__":
    main()