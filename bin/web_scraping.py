import requests
from bs4 import BeautifulSoup

data = {}

def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None


def scrap_html(html):
    if html is None:
        print("No HTML to parse")
        return None
    try:
        soup = BeautifulSoup(html, 'html.parser')
        numbers = soup.find_all('div', class_='P6K39c')
        text = soup.find_all('div', class_='mfs7Fc')
        for number, i in zip(numbers, text):
            data[i.text] = number.text
        return data

    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None


def convert_to_float(value_str):
    try:
        if value_str.endswith('M'):
            return float(value_str[:-1]) * 1e6
        elif value_str.endswith('B'):
            return float(value_str[:-1]) * 1e9
        else:
            return float(value_str)
    except ValueError as e:
        print(f"Error converting '{value_str}' to float: {e}")
        return None


def fix_data(data: dict) -> dict:
    if data is None:
        print("No data to fix")
        return None
    try:
        prev_close = data['Previous close'][1:]
        range = data['Day range'].split(" - ")
        low, high = range[0][1:], range[1][1:]
        volume = data['Avg Volume']
        data = {
            'Prev_close': float(prev_close),
            'Low': float(low),
            'High': float(high),
            'Volume': convert_to_float(volume)
        }
    except Exception as e:
        print("Error in extracting data from dict: ", e)
        return None
    return data


def webscarp(key, crypto=False):
    if crypto:
        url = f'https://www.google.com/finance/quote/{key}-USD'
    else:
        url = f'https://www.google.com/finance/quote/{key}:NASDAQ'
    html = get_html(url)
    if html is None:
        return None
    data = scrap_html(html)
    if data is None:
        return None
    data = fix_data(data)
    return data


if __name__ == "__main__":
    webscarp("NVDA", False)