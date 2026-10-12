from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import runpy
import threading
from urllib.parse import parse_qs
from unittest.mock import MagicMock

import pytest


# Import the standalone script without initializing the scripts/ML packages.
scraper = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/download_data.py'))
HEADER = scraper['HEADERS']
ROW = 'ATB;01/10/2026;3,9;4;3,8;3,95;100'


@pytest.mark.parametrize('failure_stage', ['open_form', 'download'])
def test_failed_periods_do_not_stop_other_periods_or_symbols(monkeypatch, tmp_path, capsys, failure_stage):
    playwright = pytest.importorskip('playwright.sync_api')
    manager = MagicMock()
    browser = manager.__enter__.return_value.chromium.launch.return_value
    monkeypatch.setattr(playwright, 'sync_playwright', lambda: manager)
    namespace = scraper['main'].__globals__
    periods = [(date(2024, 1, 1), date(2024, 3, 24)),
               (date(2024, 3, 25), date(2024, 6, 16)),
               (date(2024, 6, 17), date(2024, 7, 1))]
    monkeypatch.setitem(namespace, 'date_ranges', lambda start, end: iter(periods))
    monkeypatch.setattr(namespace['time'], 'sleep', lambda seconds: None)
    open_form = MagicMock(return_value=object())
    download = MagicMock(return_value=1)
    # Fail an early period in each stock, then confirm later periods still run.
    if failure_stage == 'open_form':
        open_form.side_effect = [RuntimeError('form missing'), object(), object(),
                                 RuntimeError('form missing'), object(), object()]
    else:
        download.side_effect = [playwright.TimeoutError('download timeout'), 1, 1,
                                RuntimeError('invalid CSV'), 1, 1]
    monkeypatch.setitem(namespace, 'open_form', open_form)
    monkeypatch.setitem(namespace, 'download', download)
    with pytest.raises(RuntimeError, match='Some download periods failed'):
        scraper['main'](['--date', '01-01-2024', '--symbols', 'ATB', 'BIAT',
                         '--headless', '--output-dir', str(tmp_path)])
    assert open_form.call_count == 6
    assert download.call_count == (4 if failure_stage == 'open_form' else 6)
    assert download.call_args.args[2:4] == periods[-1]
    assert download.call_args.args[-1] == 'BIAT'
    browser.close.assert_called_once()
    output = capsys.readouterr()
    assert '4 periods completed, 2 periods skipped' in output.out
    assert 'ATB: 2024-01-01 to 2024-03-24' in output.err
    assert 'BIAT: 2024-01-01 to 2024-03-24' in output.err


def test_keyboard_interrupt_still_cancels(monkeypatch):
    playwright = pytest.importorskip('playwright.sync_api')
    manager = MagicMock()
    browser = manager.__enter__.return_value.chromium.launch.return_value
    monkeypatch.setattr(playwright, 'sync_playwright', lambda: manager)
    open_form = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setitem(scraper['main'].__globals__, 'open_form', open_form)
    with pytest.raises(KeyboardInterrupt):
        scraper['main'](['--date', '01-01-2024', '--symbols', 'ATB', '--headless'])
    open_form.assert_called_once()
    browser.close.assert_called_once()


def test_date_ranges_are_contiguous_and_capped():
    periods = list(scraper['date_ranges'](date(2024, 1, 1), date(2024, 7, 1)))
    assert periods[0][0] == date(2024, 1, 1)
    assert periods[-1][1] == date(2024, 7, 1)
    assert all((end - start).days <= 83 for start, end in periods)
    assert all((periods[i + 1][0] - periods[i][1]).days == 1 for i in range(len(periods) - 1))


def test_csv_append_deduplicates_and_handles_bom(tmp_path):
    destination = tmp_path / 'ATB.csv'
    payload = f'{HEADER}\n{ROW}\n'.encode('utf-8-sig')
    assert scraper['append_csv'](payload, destination, 'ATB') == 1
    assert scraper['append_csv'](payload, destination, 'ATB') == 0
    assert destination.read_text(encoding='utf-8') == f'{HEADER}\n{ROW}\n'


def test_append_to_file_without_final_newline(tmp_path):
    destination = tmp_path / 'ATB.csv'
    destination.write_text(HEADER, encoding='utf-8')
    scraper['append_csv'](f'{HEADER}\n{ROW}'.encode(), destination, 'ATB')
    assert destination.read_text(encoding='utf-8') == f'{HEADER}\n{ROW}\n'


@pytest.mark.parametrize('content', [
    b'<html>Cloudflare: Sorry, you have been blocked</html>',
    f'{HEADER}\nATB;missing columns'.encode(),
    f'{HEADER}\n{ROW.replace("ATB", "OTHER")}'.encode(),
])
def test_invalid_response_preserves_existing_csv(tmp_path, content):
    destination = tmp_path / 'ATB.csv'
    destination.write_text(f'{HEADER}\n{ROW}\n', encoding='utf-8')
    before = destination.read_bytes()
    with pytest.raises(RuntimeError):
        scraper['append_csv'](content, destination, 'ATB')
    assert destination.read_bytes() == before


def test_form_submission_and_download_in_real_browser(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    submissions = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Set-Cookie', 'session=fixture; Path=/')
            self.end_headers()
            self.wfile.write('''<form method="post" action="/export">
                <input type="date" name="dtFrom">
                <input type="date" name="dtTo">
                <input type="hidden" name="__RequestVerificationToken" value="fixture-token">
                <input type="hidden" name="__Invariant" value="dtFrom">
                <input type="hidden" name="__Invariant" value="dtTo">
                <button type="submit">Télécharger</button>
                </form>'''.encode())

        def do_POST(self):
            fields = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode())
            submissions.append((fields, self.headers.get('Cookie')))
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Disposition', 'attachment; filename=ATB.csv')
            self.end_headers()
            self.wfile.write(f'{HEADER}\r\n{ROW}\r\n'.encode('utf-8-sig'))

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel='chrome', headless=True)
            except playwright.Error as error:
                if 'not found' in str(error) or "doesn't exist" in str(error):
                    pytest.skip('Google Chrome is not installed')
                raise
            try:
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()
                form = scraper['open_form'](page, f'http://127.0.0.1:{server.server_port}', interactive=False)
                count = scraper['download'](page, form, date(2026, 10, 1), date(2026, 10, 7), tmp_path / 'ATB.csv', 'ATB')
                assert count == 1
                assert (tmp_path / 'ATB.csv').read_text(encoding='utf-8') == f'{HEADER}\n{ROW}\n'
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    fields, cookie = submissions[0]
    assert fields['dtFrom'] == ['2026-10-01']
    assert fields['dtTo'] == ['2026-10-07']
    assert fields['__RequestVerificationToken'] == ['fixture-token']
    assert fields['__Invariant'] == ['dtFrom', 'dtTo']
    assert cookie == 'session=fixture'
