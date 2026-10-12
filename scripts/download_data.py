r"""Scrape ilboursa using a visible browser and its stock-history form.

Setup:
    .\.venv\Scripts\python.exe -m pip install playwright

Run from the project root (uses installed Google Chrome):
    python scripts/download_data.py --date 01-10-2026 --symbols ATB

Use --date all for history since 2008, or omit --symbols for all configured stocks.
For bundled Chromium: python -m playwright install chromium, then --browser chromium.
If verification is required, complete it manually in the visible browser and
press Enter in the terminal. The script does not automatically solve challenges.
"""

import argparse
import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import runpy
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load shared constants without importing utils/__init__.py and the ML stack.
_constants = runpy.run_path(str(PROJECT_ROOT / 'utils' / 'constants.py'))
RAW_DATA_DOWNLOAD_BASELINK = _constants['RAW_DATA_DOWNLOAD_BASELINK']
HEADERS = _constants['HEADERS']
SYMBOLS = _constants['SYMBOLS']


def date_ranges(start, end):
    """Yield inclusive ranges of at most 84 days, never exceeding end."""
    while start <= end:
        stop = min(start + timedelta(days=83), end)
        yield start, stop
        start = stop + timedelta(days=1)


def append_csv(content, destination, symbol):
    """Validate a downloaded CSV and append only rows not already saved."""
    rows = list(csv.reader(content.decode('utf-8-sig').splitlines(), delimiter=';'))
    expected_header = HEADERS.split(';')
    if not rows or rows[0] != expected_header:
        raise RuntimeError(f'{symbol}: expected stock CSV, received another response. No data saved.')
    records = [row for row in rows[1:] if row]
    if any(len(row) != len(expected_header) or row[0] != symbol for row in records):
        raise RuntimeError(f'{symbol}: unexpected CSV columns or stock symbol. No data saved.')

    destination = Path(destination)
    existing = set()
    has_header = destination.exists() and destination.stat().st_size > 0
    if has_header:
        with destination.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, delimiter=';')
            if next(reader, None) != expected_header:
                raise RuntimeError(f'Existing file has an unexpected header: {destination}')
            existing = {tuple(row) for row in reader}
    new_rows = []
    for row in records:
        key = tuple(row)
        if key not in existing:
            new_rows.append(row)
            existing.add(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if has_header and new_rows:
        with destination.open('rb') as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) not in (b'\n', b'\r')
    with destination.open('a', encoding='utf-8', newline='') as handle:
        if needs_newline:
            handle.write('\n')
        writer = csv.writer(handle, delimiter=';', lineterminator='\n')
        if not has_header:
            writer.writerow(expected_header)
        writer.writerows(new_rows)
    return len(new_rows)


def open_form(page, url, interactive=True):
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    form = page.locator('form').filter(has=page.locator('input[name="dtFrom"]')).first
    try:
        form.locator('input[name="dtFrom"]').wait_for(state='visible', timeout=15000)
    except PlaywrightTimeoutError as error:
        if not interactive:
            raise RuntimeError(f'Download form unavailable at {url}. Retry in visible browser mode.') from error
        print(f'Form unavailable at {url}. Browser page: {page.title()}', flush=True)
        try:
            input('Complete any verification/login in the browser, then press Enter here. Ctrl+C cancels: ')
        except EOFError as input_error:
            raise RuntimeError('Manual browser interaction requires an interactive terminal.') from input_error
        try:
            form.locator('input[name="dtFrom"]').wait_for(state='visible', timeout=15000)
        except PlaywrightTimeoutError as retry_error:
            raise RuntimeError(
                f'The form is still unavailable at {url}. The site may be blocking this browser, '
                'or the form has changed. No download was submitted.'
            ) from retry_error
    return form


def download(page, form, start, end, destination, symbol):
    for name, value in [('dtFrom', start), ('dtTo', end)]:
        field = form.locator(f'input[name="{name}"]')
        # HTML date controls require ISO dates; text controls use the site's French format.
        formatted = value.isoformat() if field.get_attribute('type') == 'date' else value.strftime('%d/%m/%Y')
        field.fill(formatted)
    with page.expect_download(timeout=3000) as pending:
        form.get_by_role('button', name='Télécharger', exact=True).click()
    result = pending.value
    failure = result.failure()
    if failure:
        raise RuntimeError(f'{symbol}: browser download failed: {failure}')
    saved = append_csv(Path(result.path()).read_bytes(), destination, symbol)
    result.delete()
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scrape ilboursa stock history with Playwright.')
    parser.add_argument('--date', required=True, help='Start date: dd-mm-yyyy or all')
    parser.add_argument('--symbols', nargs='+', choices=SYMBOLS, default=SYMBOLS,
                        metavar='SYMBOL', help='Stock symbols, e.g. ATB BIAT; defaults to all configured stocks')
    parser.add_argument('--browser', choices=['chrome', 'msedge', 'chromium'], default='chrome')
    parser.add_argument('--headless', action='store_true', help='Disable the visible browser and manual verification')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw')
    args = parser.parse_args(argv)
    try:
        start = date(2010, 1, 1) if args.date.lower() == 'all' else datetime.strptime(args.date, '%d-%m-%Y').date()
    except ValueError:
        parser.error('--date must be dd-mm-yyyy or all')
    today = date.today()
    if start > today:
        parser.error('--date must not be in the future')
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        parser.exit(1, 'Playwright is missing. Install it with: python -m pip install playwright\n')

    with sync_playwright() as playwright:
        options = {'headless': args.headless}
        if args.browser != 'chromium':
            options['channel'] = args.browser
        browser = playwright.chromium.launch(**options)
        try:
            context = browser.new_context(accept_downloads=True, locale='fr-FR')
            page = context.new_page()
            failed_periods = []
            completed_periods = 0
            for symbol in args.symbols:
                url = f'{RAW_DATA_DOWNLOAD_BASELINK}{symbol}'
                for period_start, period_end in date_ranges(start, today):
                    try:
                        form = open_form(page, url, interactive=not args.headless)
                        count = download(page, form, period_start, period_end,
                                         args.output_dir / f'{symbol}.csv', symbol)
                    except Exception as error:
                        failure = f'{symbol}: {period_start} to {period_end}: {error}'
                        failed_periods.append(failure)
                        print(f'Skipping failed period — {failure}', file=sys.stderr, flush=True)
                    else:
                        completed_periods += 1
                        print(f'{symbol}: {period_start} to {period_end}: saved {count} new rows', flush=True)
                    time.sleep(2)  # Space out requests across ranges and symbols.
            print(f'Finished: {completed_periods} periods completed, {len(failed_periods)} periods skipped.', flush=True)
            if failed_periods:
                print('Failed periods:', file=sys.stderr)
                for failure in failed_periods:
                    print(f'  {failure}', file=sys.stderr)
        finally:
            browser.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Cancelled.')
    except Exception as error:
        sys.exit(str(error))
