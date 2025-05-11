import json
import sqlite3
import logging
from .spam_detector import SpamDetector

logger = logging.getLogger('SpamMetrics')
logging.basicConfig(level=logging.INFO)

class SpamMetrics:
    def __init__(self, db_path):
        self.db_path = db_path
        self.spam_detector = SpamDetector(db_path)
    
    def get_metrics(self, video_id=None):
        """Get spam detection metrics, optionally filtered by video_id"""
        # Get model status and accuracy
        model_status = "untrained"
        accuracy = None
        
        if self.spam_detector.is_trained:
            model_status = "trained"
            accuracy = self.spam_detector.get_accuracy()
        
        # Get training sample counts
        spam_samples, ham_samples = self._get_sample_counts(video_id)
        
        # Get top features if model is trained
        top_features = {}
        if self.spam_detector.is_trained and hasattr(self.spam_detector.model, 'feature_importances_'):
            top_features = self._get_top_features()
        
        return {
            "model_status": model_status,
            "accuracy": accuracy,
            "spam_samples": spam_samples,
            "ham_samples": ham_samples,
            "total_samples": spam_samples + ham_samples,
            "top_features": top_features
        }
    
    def _get_sample_counts(self, video_id=None):
        """Get counts of spam and ham samples in the training data"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT COUNT(*) as count, is_spam FROM comments WHERE is_spam IS NOT NULL"
            params = []
            if video_id:
                query += " AND video_id = ?"
                params.append(video_id)
            query += " GROUP BY is_spam"
            cursor.execute(query, params)
            results = cursor.fetchall()
            spam_count = 0
            ham_count = 0
            for row in results:
                if row['is_spam']:
                    spam_count = row['count']
                else:
                    ham_count = row['count']
            return spam_count, ham_count
        except Exception as e:
            logger.error(f"Error in _get_sample_counts: {e}")
            return 0, 0
        finally:
            try:
                conn.close()
            except Exception:
                pass
    
    def _get_top_features(self, limit=10):
        """Get the top spam features by importance"""
        if not hasattr(self.spam_detector, 'model') or not hasattr(self.spam_detector.model, 'feature_importances_'):
            return {}
        
        # Get feature importances and feature names
        importances = self.spam_detector.model.feature_importances_
        feature_names = self.spam_detector.vectorizer.get_feature_names_out() if hasattr(self.spam_detector.vectorizer, 'get_feature_names_out') else []
        
        if len(feature_names) == 0:
            return {}
        
        # Create a dictionary of feature name to importance
        feature_importance = {}
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                feature_name = feature_names[i]
                feature_importance[feature_name] = float(importance)
        
        # Sort by importance and take top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:limit])
        
        return top_features
