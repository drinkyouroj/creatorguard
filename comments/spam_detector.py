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
            # Try to load both model and vectorizer
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                try:
                    self.model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                    
                    # Verify model and vectorizer are valid
                    if not isinstance(self.model, RandomForestClassifier) or \
                       not isinstance(self.vectorizer, TfidfVectorizer):
                        raise ValueError("Invalid model or vectorizer type")
                    
                    logger.info("✅ Loaded existing spam detection model")
                    return
                except (EOFError, ValueError) as e:
                    logger.warning(f"Failed to load model files: {str(e)}")
                    # Delete corrupted files
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(vectorizer_path):
                        os.remove(vectorizer_path)
            
            # Initialize new model if loading failed or files don't exist
            logger.info("🆕 Initializing new spam detection model")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=1000)
            
        except Exception as e:
            logger.error(f"❌ Error loading/initializing model: {str(e)}", exc_info=True)
            raise

    def save_model(self, metrics=None):
        """Save model and update database with version info."""
        conn = None
        try:
            # Save model files
            model_dir = 'models'
            model_path = os.path.join(model_dir, 'spam_model.joblib')
            vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
            
            # Create backup of existing files
            for path in [model_path, vectorizer_path]:
                if os.path.exists(path):
                    backup_path = f"{path}.bak"
                    os.rename(path, backup_path)
            
            try:
                # Save new model files
                joblib.dump(self.model, model_path)
                joblib.dump(self.vectorizer, vectorizer_path)
                
                # If save successful, remove backups
                for path in [model_path, vectorizer_path]:
                    backup_path = f"{path}.bak"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                
                # Update database with model version
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
                parameters = json.dumps(self.model.get_params())
                metrics_json = json.dumps(metrics) if metrics else '{}'
                
                cursor.execute("""
                    INSERT INTO model_versions (
                        model_type, version, parameters, metrics, created_at
                    ) VALUES (?, ?, ?, ?, datetime('now'))
                """, ('spam', version, parameters, metrics_json))
                
                conn.commit()
                logger.info(f"✅ Saved model version {version}")
                
            except Exception as e:
                # Restore backups if save failed
                logger.error(f"❌ Failed to save model: {str(e)}")
                if conn:
                    conn.rollback()
                for path in [model_path, vectorizer_path]:
                    backup_path = f"{path}.bak"
                    if os.path.exists(backup_path):
                        if os.path.exists(path):
                            os.remove(path)
                        os.rename(backup_path, path)
                raise
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    def predict_spam(self, text):
        """Predict if text is spam and return probability."""
        try:
            # Check if model is trained
            if not hasattr(self.model, 'estimators_'):
                logger.warning("Model not trained yet, returning default prediction")
                return {
                    'is_spam': False,
                    'confidence': 0.5,
                    'spam_score': 0.5
                }

            # Transform text to feature vector
            try:
                features = self.vectorizer.transform([text])
            except Exception as e:
                logger.error(f"❌ Error transforming text: {str(e)}")
                # Retrain the model to update the vectorizer
                self.train(retrain=True)
                features = self.vectorizer.transform([text])

            # Get prediction
            prediction = self.model.predict(features)[0]
            confidence = np.max(self.model.predict_proba(features))
            
            return {
                'is_spam': bool(prediction),
                'confidence': float(confidence),
                'spam_score': float(confidence) if prediction else 1.0 - float(confidence)
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting spam: {str(e)}")
            return {
                'is_spam': False,
                'confidence': 0.5,
                'spam_score': 0.5
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
            if not isinstance(is_spam, bool):
                logger.error(f"[SPAM] Invalid is_spam value: {is_spam}, must be boolean")
                return False
                
            logger.info(f"[SPAM] SpamDetector marking comment {comment_id} as spam={is_spam}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get comment text and verify comment exists
            logger.info(f"[SPAM] Looking up comment {comment_id} in database")
            cursor.execute("SELECT comment_id, text FROM comments WHERE comment_id = ?", (comment_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"[SPAM] Comment {comment_id} not found in database")
                return False
            
            comment_id = result[0]
            text = result[1]
            logger.info(f"[SPAM] Found comment {comment_id} with text: {text[:50]}...")
            
            try:
                # Add to training data
                logger.info(f"[SPAM] Adding comment {comment_id} to spam_training")
                cursor.execute("""
                    INSERT OR REPLACE INTO spam_training (
                        comment_id, text, is_spam, confidence, trained_at
                    ) VALUES (?, ?, ?, ?, datetime('now'))
                """, (comment_id, text, is_spam, confidence))
                
                # Update comment status
                logger.info(f"[SPAM] Updating comment {comment_id} status in comments table")
                cursor.execute("""
                    UPDATE comments 
                    SET is_spam = ?, spam_score = ?, updated_at = datetime('now')
                    WHERE comment_id = ?
                """, (is_spam, 1.0 if is_spam else 0.0, comment_id))
                
                # Verify the update worked
                cursor.execute("SELECT changes()")
                if cursor.fetchone()[0] == 0:
                    logger.error(f"[SPAM] Failed to update comment {comment_id} in database")
                    conn.rollback()
                    return False
                
                conn.commit()
                logger.info(f"[SPAM] Successfully marked comment {comment_id} as {'spam' if is_spam else 'not spam'}")
                
                # Retrain model if we have enough new samples
                logger.info("[SPAM] Checking for untrained samples")
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM spam_training
                    WHERE trained_at IS NULL
                """)
                
                untrained_count = cursor.fetchone()[0]
                if untrained_count >= 10:
                    self.train()
                
                logger.info(f"✅ Marked comment {comment_id} as {'spam' if is_spam else 'not spam'}")
                return True
                
            except Exception as e:
                conn.rollback()
                raise
            
        except Exception as e:
            logger.error(f"❌ Error marking comment as spam: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")

    def calculate_metrics(self):
        """Calculate current model metrics."""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'vectorizer'):
                logger.info("🆕 Initializing new spam detection model")
                self.model = RandomForestClassifier()
                self.vectorizer = TfidfVectorizer()
                logger.warning("No metrics available - model not initialized")
                return None

            # Get all training data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT text, is_spam
                FROM spam_training
                WHERE trained_at IS NOT NULL
            """)
            results = cursor.fetchall()
            conn.close()

            if not results:
                logger.warning("No training data available yet")
                return None

            # Check if model is trained
            if not hasattr(self.model, 'estimators_'):
                logger.warning("Model not trained yet")
                return {
                    'accuracy': None,
                    'top_features': {},
                    'total_samples': len(results),
                    'spam_samples': sum(1 for r in results if r[1]),
                    'ham_samples': sum(1 for r in results if not r[1]),
                    'model_status': 'untrained'
                }
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
