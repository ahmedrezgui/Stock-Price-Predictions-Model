from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import requests
from bs4 import BeautifulSoup
import os


from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.getcwd()) / '..'))
from utils import RAW_DATA_DOWNLOAD_BASELINK, HEADERS, SYMBOLS


class CloudflareBlockedError(RuntimeError):
    """The server returned a security block or challenge instead of stock data."""


def check_response(response):
    body = response.text.lower()
    if (response.headers.get('cf-mitigated') == 'challenge'
            or 'cf-chl-' in body
            or ('cloudflare' in body and 'sorry, you have been blocked' in body)):
        ray_id = response.headers.get('cf-ray')
        if not ray_id:
            soup = BeautifulSoup(response.text, 'html.parser')
            label = soup.find(string=lambda text: text and 'Cloudflare Ray ID:' in text)
            value = label.parent.find('strong') if label else None
            ray_id = value.get_text(strip=True) if value else 'unavailable'
        raise CloudflareBlockedError(
            f"Cloudflare blocked or challenged {response.url} "
            f"(HTTP {response.status_code}, Ray ID: {ray_id}). "
            "The download form/CSV was not received. Contact ilboursa with this "
            "Ray ID and request details to arrange automated access."
        )
    response.raise_for_status()


def get_dates(start):
    """
    Converts start date from string format to date object
    If start date is equal to current date ,it returns todays date value in start_date and end_date
    input:
        start: start date in string format "dd-mm-yyyy"
    Returns:
        start_date: start date as date object
        end_date: date after 28*3-1 = 83 days period as date object (taking into consideration feb = 28 days)

    """
    today=datetime.today().date()
    if datetime.strptime(start, "%d-%m-%Y").date() == today:
        start_date= end_date= today
    else:
        start_date = datetime.strptime(start, "%d-%m-%Y").date()
        end_date = start_date + relativedelta(days= 83)
    
    return start_date, end_date

def update_dates(date):
    """
    Updates the date range 
    input:
        date: date object
    Returns:
        start_date: next day date from input as date object
        end_date: date after start_date with 83 days period as date object

    """
    start_date= date + relativedelta(days= 1)
    end_date=start_date+relativedelta(days= 83)
    return start_date,end_date

def download(start_date, end_date ,cookies, token, session, link, fileName):
    # Extract cookies explicitly
    cookies = session.cookies.get_dict()

    # Prepare POST data (form fields)
    # A list preserves both __Invariant fields; a dict discards the first one.
    data = [
        ('dtFrom', start_date),
        ('__Invariant', 'dtFrom'),
        ('dtTo', end_date),
        ('__Invariant', 'dtTo'),
        ('__RequestVerificationToken', token),
    ]

    # Prepare headers to mimic a real browser
    post_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.ilboursa.com",
        "Referer": link,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    }

    # Make POST request with data + headers + EXPLICIT cookies
    response_post = session.post(link, data=data, headers=post_headers, cookies=cookies, timeout=30)
    check_response(response_post)

    lines = response_post.content.decode('utf-8-sig').splitlines()
    if not lines or lines[0] != HEADERS:
        raise RuntimeError(
            f"Expected CSV from {link}, received HTTP {response_post.status_code} "
            f"with Content-Type {response_post.headers.get('Content-Type', 'unknown')}. "
            "The response may be an error, login page, or no-data page; no file was written."
        )
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{fileName}"
    with open(file_path, 'a', encoding='utf-8') as f:
        if os.stat(file_path).st_size == 0:
            print(f"{file_path} created successfully")
            f.write(f'{HEADERS}\n')
        for line in lines[1:]:
            if line:
                f.write(f'{line}\n')

def main(date):
    today=datetime.today().date()

    for symbol in SYMBOLS:

        start_date, end_date= get_dates(date)

        # Create a session to manage cookies automatically
        session = requests.Session()

        # First GET request to load page and retrieve token + cookies
        url_get = f'{RAW_DATA_DOWNLOAD_BASELINK}{symbol}'
        response_get = session.get(url_get, timeout=30)
        check_response(response_get)

        # Parse the __RequestVerificationToken from the HTML
        soup = BeautifulSoup(response_get.text, "html.parser")
        token_input = soup.find('input', {'name': '__RequestVerificationToken'})
        if token_input is None or not token_input.get('value'):
            raise RuntimeError(
                f"Download form token missing at {url_get} (HTTP {response_get.status_code}). "
                "The page may require login or its form may have changed."
            )
        token = token_input['value']

        cookies = session.cookies.get_dict()

        fileName = f'{symbol}.csv'

        while  today >= start_date:
            download(start_date,end_date,cookies,token,session,url_get,fileName)
            #update the start and end dates
            start_date, end_date = update_dates(end_date)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Download automation for stock data.")
    parser.add_argument("--date", required=True, help="Start date in dd-mm-YYYY format")
    args = parser.parse_args()
    if args.date.lower() == "all":
        start_date_str = "01-01-2008"
    else:
        start_date_str = args.date

    main(start_date_str)

