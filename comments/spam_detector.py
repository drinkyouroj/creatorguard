import os
import sqlite3
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import re
from collections import Counter
from utils.logger import setup_logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set up logger for this module
logger = setup_logger(__name__)

class SpamDetector:
    def __init__(self, db_path='creatorguard.db'):
        """Initialize spam detector with database connection and model loading."""
        self.db_path = db_path
        self.model = None
        self.vectorizer = None
        
        # Create models directory if it doesn't exist
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or initialize model
        self.load_latest_model()

    def extract_features(self, text):
        """Extract spam-related features from text."""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'mention_count': len(re.findall(r'@\w+', text)),
            'hashtag_count': len(re.findall(r'#\w+', text)),
            'emoji_count': len(re.findall(r'[\U0001F300-\U0001F9FF]', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
            'has_email': 1 if re.search(r'[^@]+@[^@]+\.[^@]+', text) else 0,
            'spam_word_ratio': self._calculate_spam_word_ratio(text)
        }
        return features

    def _calculate_spam_word_ratio(self, text):
        """Calculate ratio of potential spam words in text."""
        spam_indicators = {
            'free', 'win', 'winner', 'won', 'prize', 'money', 'cash', 'gift', 
            'click', 'subscribe', 'offer', 'limited', 'hurry', 'discount', 'deal',
            'guarantee', 'instant', 'now', 'urgent', 'verify', 'verified', 'check',
            'congratulations', 'selected', 'lottery', 'promotion'
        }
        
        words = set(word.lower() for word in word_tokenize(text))
        if not words:
            return 0
        return len(words.intersection(spam_indicators)) / len(words)

    def load_latest_model(self):
        """Load existing model or initialize a new one."""
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'spam_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("✅ Loaded existing spam detection model")
        except FileNotFoundError:
            logger.info("🆕 Initializing new spam detection model")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=1000)

    def save_model(self, metrics=None):
        """Save model and update database with version info."""
        # Save model files
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'spam_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Update database with model version
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        parameters = json.dumps(self.model.get_params())
        metrics_json = json.dumps(metrics) if metrics else '{}'
        
        cursor.execute("""
            INSERT INTO model_versions (model_type, version, metrics, parameters)
            VALUES (?, ?, ?, ?)
        """, ('spam', version, metrics_json, parameters))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Saved spam detection model version {version}")

    def predict_spam(self, text):
        """Predict if text is spam and return probability."""
        try:
            if not self.model or not self.vectorizer:
                logger.warning("Model not initialized, loading latest model...")
                self.load_latest_model()
            
            # Transform text using vectorizer
            features = self.vectorizer.transform([text])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0][1]
            
            return {
                'is_spam': bool(prediction),
                'probability': float(probability),
                'features': self.extract_features(text)
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting spam: {str(e)}")
            # Return safe default
            return {
                'is_spam': False,
                'probability': 0.0,
                'features': {}
            }

    def train(self, retrain=False):
        """Train the spam detection model using collected data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get training data
        cursor.execute("""
            SELECT st.text, st.features, st.is_spam, st.confidence
            FROM spam_training st
            WHERE st.trained_at > (
                SELECT COALESCE(MAX(created_at), '1970-01-01')
                FROM model_versions
                WHERE model_type = 'spam'
            )
            OR ?
        """, (retrain,))
        
        training_data = cursor.fetchall()
        
        if not training_data and not retrain:
            logger.info("No new training data available")
            return
        
        # Prepare features and labels
        texts = [row[0] for row in training_data]
        labels = [row[2] for row in training_data]
        confidences = [row[3] for row in training_data]
        
        # Extract text features using TF-IDF
        if retrain:
            self.vectorizer = TfidfVectorizer(max_features=1000)
        text_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Extract and combine other features
        other_features = []
        for text in texts:
            features = self.extract_features(text)
            other_features.append(list(features.values()))
        
        # Combine all features
        X = np.hstack((text_features, np.array(other_features)))
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if retrain:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
        
        # Save model and metrics
        self.save_model(metrics)
        
        conn.close()
        return metrics

    def mark_as_spam(self, comment_id, is_spam, confidence=1.0):
        """Mark a comment as spam/not spam for training."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get comment text
            cursor.execute("SELECT text FROM comments WHERE comment_id = ?", (comment_id,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Comment {comment_id} not found")
            
            text = result[0]
            
            # Add to training data
            cursor.execute("""
                INSERT OR REPLACE INTO spam_training (comment_id, is_spam, confidence, created_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (comment_id, is_spam, confidence))
            
            conn.commit()
            
            # Retrain model if we have enough new samples
            cursor.execute("""
                SELECT COUNT(*)
                FROM spam_training
                WHERE trained_at IS NULL
            """)
            
            untrained_count = cursor.fetchone()[0]
            if untrained_count >= 10:
                self.train()
            
            return True
            
            logger.info(f"✅ Marked comment {comment_id} as {'spam' if is_spam else 'not spam'}")
            
        except Exception as e:
            logger.error(f"❌ Error marking comment as spam: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
            conn.close()

    def calculate_metrics(self):
        """Calculate current model performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all labeled data
            cursor.execute("""
                SELECT c.text, st.is_spam
                FROM spam_training st
                JOIN comments c ON c.comment_id = st.comment_id
                ORDER BY st.trained_at DESC
                LIMIT 1000
            """)
            
            data = cursor.fetchall()
            if not data:
                logger.warning("No labeled data available for metrics calculation")
                return None
                
            texts, labels = zip(*data)
            
            # If model or vectorizer not initialized, return None
            if not self.model or not self.vectorizer:
                logger.warning("Model not initialized")
                return None
            
            try:
                # Transform texts using the same vectorizer that trained the model
                features = self.vectorizer.transform(texts)
                
                # Get predictions
                predictions = self.model.predict(features)
                
                # Calculate metrics
                metrics = {
                    'accuracy': float(accuracy_score(labels, predictions)),
                    'precision': float(precision_score(labels, predictions)),
                    'recall': float(recall_score(labels, predictions)),
                    'f1': float(f1_score(labels, predictions)),
                    'total_samples': len(labels),
                    'spam_ratio': sum(labels) / len(labels),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Save metrics
                cursor.execute("""
                    INSERT INTO model_metrics (
                        model_type, metrics, created_at
                    ) VALUES (?, ?, ?)
                """, ('spam', json.dumps(metrics), metrics['timestamp']))
                
                conn.commit()
                logger.info(f"✅ Calculated metrics: accuracy={metrics['accuracy']:.2f}, precision={metrics['precision']:.2f}")
                return metrics
                
            except ValueError as e:
                logger.error(f"❌ Error during prediction: {str(e)}")
                # Only retrain once
                if "retraining" not in str(e):
                    logger.warning("Attempting one-time retrain...")
                    self.train(retrain=True)
                    return self.calculate_metrics()
                return None
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {str(e)}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    def get_metrics_history(self, days=30):
        """Get historical metrics for the past N days."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metrics, created_at
                FROM model_metrics
                WHERE model_type = 'spam'
                AND created_at >= datetime('now', ?)
                ORDER BY created_at ASC
            """, (f'-{days} days',))
            
            return [
                {
                    'metrics': json.loads(row[0]),
                    'created_at': row[1]
                }
                for row in cursor.fetchall()
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics history: {str(e)}", exc_info=True)
            return []
        finally:
            conn.close()

    def get_spam_trends(self, days=30):
        """Get spam trends over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    date(timestamp) as day,
                    COUNT(*) as total_comments,
                    SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as spam_comments
                FROM comments
                WHERE timestamp >= datetime('now', ?)
                GROUP BY date(timestamp)
                ORDER BY day ASC
            """, (f'-{days} days',))
            
            return [
                {
                    'date': row[0],
                    'total': row[1],
                    'spam': row[2],
                    'ratio': row[2] / row[1] if row[1] > 0 else 0
                }
                for row in cursor.fetchall()
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting spam trends: {e}")
            return []
        finally:
            conn.close()

if __name__ == '__main__':
    # Example usage
    detector = SpamDetector()
    
    # Train model if needed
    detector.train()
    
    # Test prediction
    test_comment = "FREE IPHONE! Click here to claim your prize! www.scam.com"
    result = detector.predict_spam(test_comment)
    logger.info(f"Spam prediction: {result}")
