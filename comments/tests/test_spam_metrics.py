import pytest
from unittest.mock import patch, MagicMock
from comments.spam_metrics import SpamMetrics

@patch('comments.spam_metrics.SpamDetector')
def test_get_metrics_untrained(mock_detector):
    mock_detector.return_value.is_trained = False
    metrics = SpamMetrics(':memory:').get_metrics()
    assert metrics['model_status'] == 'untrained'
    assert metrics['accuracy'] is None

@patch('comments.spam_metrics.SpamDetector')
@patch('comments.spam_metrics.sqlite3.connect')
def test_get_metrics_trained(mock_connect, mock_detector):
    mock_detector.return_value.is_trained = True
    mock_detector.return_value.get_accuracy.return_value = 0.99
    mock_detector.return_value.model = MagicMock(feature_importances_=[0.5,0.5])
    mock_detector.return_value.vectorizer = MagicMock(get_feature_names_out=lambda: ['a','b'])
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{'is_spam':1,'count':5},{'is_spam':0,'count':10}]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    metrics = SpamMetrics(':memory:').get_metrics()
    assert metrics['model_status'] == 'trained'
    assert metrics['spam_samples'] == 5
    assert metrics['ham_samples'] == 10
    assert 'a' in metrics['top_features']
    assert 'b' in metrics['top_features']
