import pytest
from unittest.mock import patch, MagicMock
from comments.analyze_comments import CommentAnalyzer

@patch('comments.analyze_comments.SpamDetector')
@patch('comments.analyze_comments.SentimentIntensityAnalyzer')
@patch('comments.analyze_comments.pipeline')
def test_analyze_comment_handles_error(mock_pipeline, mock_sia, mock_detector):
    mock_sia.return_value.polarity_scores.side_effect = Exception('fail')
    analyzer = CommentAnalyzer(db_path=':memory:')
    result = analyzer.analyze_comment('bad comment')
    assert result is None

@patch('comments.analyze_comments.SpamDetector')
@patch('comments.analyze_comments.SentimentIntensityAnalyzer')
@patch('comments.analyze_comments.pipeline')
def test_analyze_comment_success(mock_pipeline, mock_sia, mock_detector):
    mock_sia.return_value.polarity_scores.return_value = {'compound': 0.5}
    mock_pipeline.return_value = lambda *args, **kwargs: {'toxicity': 0.1}
    mock_detector.return_value.predict_spam.return_value = {'spam_score':0.1,'is_spam':False,'spam_features':{}}
    analyzer = CommentAnalyzer(db_path=':memory:')
    result = analyzer.analyze_comment('good comment')
    assert result['sentiment'] == 'positive' or result['sentiment'] == 'neutral' or result['sentiment'] == 'negative'
    assert 'classification' in result
    assert 'spam_score' in result
