import pytest
import requests
from datetime import datetime, timedelta
from unittest import mock

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.download_data import get_dates, update_dates, download, main, check_response, CloudflareBlockedError


def response(status=200, body=b'', headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result.url = 'https://www.ilboursa.com/marches/download/ATB'
    result.headers.update(headers or {})
    return result


@pytest.mark.parametrize('status', [200, 403])
def test_cloudflare_block_reports_ray_id(status):
    blocked = response(status, b'<h1>Sorry, you have been blocked</h1>Cloudflare', {'cf-ray': 'test-ray'})
    with pytest.raises(CloudflareBlockedError, match=f'HTTP {status}, Ray ID: test-ray'):
        check_response(blocked)


def test_cloudflare_ray_id_from_html():
    blocked = response(403, b'<h1>Sorry, you have been blocked</h1><span>Cloudflare Ray ID: <strong>body-ray</strong></span>')
    with pytest.raises(CloudflareBlockedError, match='body-ray'):
        check_response(blocked)


def test_challenge_header():
    with pytest.raises(CloudflareBlockedError):
        check_response(response(403, b'Please wait', {'cf-mitigated': 'challenge'}))


def test_blocked_get_stops_before_post():
    session = mock.Mock()
    session.get.return_value = response(403, b'Cloudflare: Sorry, you have been blocked')
    with mock.patch('scripts.download_data.requests.Session', return_value=session):
        with pytest.raises(CloudflareBlockedError):
            main('01-01-2024')
    session.get.assert_called_once()
    session.post.assert_not_called()


@pytest.mark.parametrize('blocked', [False, True])
def test_html_download_does_not_modify_existing_csv(tmp_path, monkeypatch, blocked):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'data' / 'raw'
    folder.mkdir(parents=True)
    destination = folder / 'ATB.csv'
    destination.write_text('existing data', encoding='utf-8')
    session = mock.Mock()
    body = b'Cloudflare: Sorry, you have been blocked' if blocked else b'<html>No data</html>'
    session.post.return_value = response(200, body)
    with pytest.raises(RuntimeError):
        download('2024-01-01', '2024-03-24', {}, 'token', session, 'https://www.ilboursa.com/marches/download/ATB', 'ATB.csv')
    assert destination.read_text(encoding='utf-8') == 'existing data'


def test_csv_download_preserves_repeated_form_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = mock.Mock()
    csv = 'symbole;date;ouverture;haut;bas;cloture;volume\nATB;01/01/2024;3;4;2;3;100\n'
    session.post.return_value = response(200, csv.encode('utf-8-sig'))
    download('2024-01-01', '2024-03-24', {}, 'token', session, 'https://www.ilboursa.com/marches/download/ATB', 'ATB.csv')
    assert (tmp_path / 'data/raw/ATB.csv').read_text(encoding='utf-8') == csv
    fields = session.post.call_args.kwargs['data']
    assert [value for key, value in fields if key == '__Invariant'] == ['dtFrom', 'dtTo']

# -------------- Fixtures ----------------
@pytest.fixture
def today_date():
    return datetime.today().date()

@pytest.fixture
def session_mock():
    return mock.Mock()

@pytest.fixture
def dummy_response():
    mock_resp = requests.Response()
    mock_resp.url = "https://www.ilboursa.com/marches/download/DUMMY"
    mock_resp.status_code = 200
    mock_resp._content = b"symbole;date;ouverture;haut;bas;cloture;volume\r\nSYM1;01-01-2024;100;110;90;105;1000"
    return mock_resp

# -------------- Tests for get_dates ----------------
class TestGetDates:

    def test_get_dates_today(self, today_date):
        """Test when start date is today's date."""
        start_str = today_date.strftime("%d-%m-%Y")
        start, end = get_dates(start_str)
        assert start == today_date
        assert end == today_date

    @pytest.mark.parametrize("input_date", ["01-01-2024", "15-08-2023"])
    def test_get_dates_past_dates(self, input_date):
        """Test normal case with past dates."""
        expected_start = datetime.strptime(input_date, "%d-%m-%Y").date()
        expected_end = expected_start + timedelta(days=83)
        start, end = get_dates(input_date)
        assert start == expected_start
        assert end == expected_end

# -------------- Tests for update_dates ----------------
class TestUpdateDates:

    def test_update_dates_normal(self):
        """Test that update_dates shifts by +1 and +84 days."""
        original_date = datetime(2024, 1, 1).date()
        start, end = update_dates(original_date)
        assert start == original_date + timedelta(days=1)
        assert end == original_date + timedelta(days=84)

# -------------- Tests for download_data ----------------
class TestDownloadData:

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists", return_value=False)
    @mock.patch("os.stat")  # <--- Mock os.stat too
    def test_download_successful(self, mock_stat, mock_exists, mock_makedirs, mock_open_file, session_mock, dummy_response):
        """Test successful download and file creation."""
        session_mock.post.return_value = dummy_response
        # Mock st_size attribute to simulate empty file (size 0)
        mock_stat.return_value.st_size = 0

        download(
            "2024-01-01",
            "2024-03-24",
            cookies={},
            token="dummy_token",
            session=session_mock,
            link="https://www.ilboursa.com/marches/download/DUMMY",
            fileName="DUMMY.csv"
        )

    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists", return_value=True)
    def test_download_failure(self, mock_exists, mock_makedirs, session_mock):
        """Test failed download raises Exception."""
        failed_response = requests.Response()
        failed_response.status_code = 500
        failed_response._content = b"Server error"
        session_mock.post.return_value = failed_response

        with pytest.raises(requests.HTTPError, match="500"):
            download(
                "2024-01-01",
                "2024-03-24",
                cookies={},
                token="dummy_token",
                session=session_mock,
                link="https://www.ilboursa.com/marches/download/DUMMY",
                fileName="DUMMY.csv"
            )

