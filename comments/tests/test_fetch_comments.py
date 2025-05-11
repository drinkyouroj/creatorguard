import pytest
from unittest.mock import patch, MagicMock
from comments.fetch_comments import YouTubeCommentFetcher

@pytest.fixture
def fetcher():
    with patch('comments.fetch_comments.build') as mock_build:
        mock_build.return_value = MagicMock()
        return YouTubeCommentFetcher(db_path=':memory:')

def test_missing_api_key(monkeypatch):
    monkeypatch.delenv('YOUTUBE_API_KEY', raising=False)
    with pytest.raises(ValueError):
        YouTubeCommentFetcher(db_path=':memory:')

def test_parse_youtube_timestamp(fetcher):
    ts = '2022-01-01T12:00:00Z'
    assert fetcher.parse_youtube_timestamp(ts) == '2022-01-01 12:00:00'
    assert isinstance(fetcher.parse_youtube_timestamp(None), str)

@patch('comments.fetch_comments.sqlite3.connect')
def test_fetch_video_metadata_handles_error(mock_connect, fetcher):
    fetcher.youtube.videos.return_value.list.return_value.execute.side_effect = Exception('fail')
    assert fetcher.fetch_video_metadata('badid') is None

@patch('comments.fetch_comments.sqlite3.connect')
def test_fetch_comments_handles_db_error(mock_connect, fetcher):
    fetcher.youtube.videos.return_value.list.return_value.execute.return_value = {'items': [{}]}
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.side_effect = Exception('db fail')
    assert fetcher.fetch_comments('vid') == 0
