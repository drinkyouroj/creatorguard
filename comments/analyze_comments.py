import os
import sqlite3
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import Counter
import re
import sys
import json
from comments.spam_detector import SpamDetector

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger, log_error, log_warning, log_info

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        """Initialize the comment analyzer."""
        self.db_path = db_path
        try:
            self.logger = setup_logger('comment_analyzer')
        except Exception:
            import logging
            self.logger = logging.getLogger('comment_analyzer')
            logging.basicConfig(level=logging.INFO)
        self.spam_detector = SpamDetector(db_path)
        
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize VADER")
            raise
            
        try:
            # Load toxicity classifier
            self.toxicity_pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
            self.logger.info("Initialized toxic-bert model")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize toxicity model")
            self.toxicity_pipeline = None

    def analyze_comment(self, text):
        """Analyze a single comment for toxicity and spam."""
        try:
            # Get toxicity scores
            toxicity_scores = self.get_toxicity_scores(text)
            
            # Get sentiment scores
            sentiment_scores = self.sia.polarity_scores(text)
            sentiment = 'positive' if sentiment_scores['compound'] > 0.05 else 'negative' if sentiment_scores['compound'] < -0.05 else 'neutral'
            
            # Get spam prediction
            spam_result = self.spam_detector.predict_spam(text)
            
            # Determine content classification (spam/toxic/questionable/safe)
            content_classification = self.determine_classification(
                toxicity_scores.get('toxicity', 0),
                spam_result['spam_score'],
                sentiment_scores['compound']
            )
            
            # Create a combined classification that includes both content and sentiment
            # Format: "content_classification:sentiment"
            combined_classification = f"{content_classification}:{sentiment}"
            
            self.logger.info(f"Classified comment as '{combined_classification}': "
                           f"toxicity={toxicity_scores.get('toxicity', 0):.2f}, "
                           f"spam={spam_result['spam_score']:.2f}, "
                           f"sentiment={sentiment_scores['compound']:.2f}")
            
            return {
                'classification': content_classification,  # Keep original field for backward compatibility
                'combined_classification': combined_classification,  # New field with combined info
                'toxicity_score': toxicity_scores.get('toxicity', 0),
                'toxicity_details': json.dumps(toxicity_scores),
                'sentiment': sentiment,
                'sentiment_scores': json.dumps(sentiment_scores),
                'spam_score': spam_result['spam_score'],
                'is_spam': spam_result['is_spam'],
                'spam_features': json.dumps(spam_result['spam_features'])
            }
            
        except Exception as e:
            log_error(self.logger, e, f"Failed to analyze comment: {text[:100]}...")
            return None

    def determine_classification(self, toxicity_score, spam_score, sentiment_score=None):
        """Determine comment classification based on toxicity and spam scores.
        
        Classification categories:
        - spam: Comments identified as spam with high confidence
        - toxic: Non-spam comments with high toxicity
        - questionable: Comments with moderate toxicity or spam indicators
        - safe: Comments with low toxicity and spam scores
        
        Sentiment is handled separately and not part of the classification.
        """
        # Use more conservative thresholds to avoid misclassification
        if spam_score >= 0.85:  # Only very high confidence spam
            return 'spam'
        elif toxicity_score >= 0.8:  # High toxicity
            return 'toxic'
        elif toxicity_score >= 0.5 or spam_score >= 0.6:  # Moderate issues
            return 'questionable'
        else:
            return 'safe'

    def analyze_comments(self, video_id):
        """Analyze all comments for a video."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get unanalyzed comments
            cursor.execute("""
                SELECT comment_id, text 
                FROM comments 
                WHERE video_id = ? 
                AND (classification IS NULL OR spam_score IS NULL)
            """, (video_id,))
            
            comments = cursor.fetchall()
            total = len(comments)
            processed = 0
            errors = 0
            
            for comment_id, text in comments:
                try:
                    # Analyze comment
                    results = self.analyze_comment(text)
                    if not results:
                        errors += 1
                        continue
                    
                    # Update database
                    # Store both the content classification and sentiment information
                    cursor.execute("""
                        UPDATE comments 
                        SET classification = ?,
                            toxicity_score = ?,
                            toxicity_details = ?,
                            sentiment_scores = ?,
                            spam_score = ?,
                            is_spam = ?,
                            spam_features = ?
                        WHERE comment_id = ?
                    """, (
                        results['combined_classification'],  # Use combined classification
                        results['toxicity_score'],
                        results['toxicity_details'],
                        json.dumps({'sentiment': results['sentiment'], 'scores': results['sentiment_scores']}),
                        results['spam_score'],
                        results['is_spam'],
                        results['spam_features'],
                        comment_id
                    ))
                    
                    processed += 1
                    
                except Exception as e:
                    log_error(self.logger, e, f"Failed to analyze comment {comment_id}")
                    errors += 1
            
            conn.commit()
            log_info(self.logger, f"Analysis complete: {processed} processed, {errors} errors")
            
        except Exception as e:
            log_error(self.logger, e, "Failed to analyze comments")
        finally:
            conn.close()

    def mark_comment_as_spam(self, comment_id, is_spam):
        """Mark a comment as spam/not spam and use it for training."""
        conn = None
        try:
            self.logger.info(f"[SPAM] Analyzer received request to mark comment {comment_id} as spam={is_spam}")
            
            # First verify the comment exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            self.logger.info(f"[SPAM] Looking up comment {comment_id} in database")
            cursor.execute("SELECT comment_id, text FROM comments WHERE comment_id = ?", (comment_id,))
            result = cursor.fetchone()
            if not result:
                self.logger.error(f"[SPAM] Comment {comment_id} not found in database")
                return {'status': 'error', 'error': f'Comment {comment_id} not found'}
            
            comment_id = result[0]
            text = result[1]
            self.logger.info(f"[SPAM] Found comment {comment_id}: {text[:50]}...")
            
            try:
                # Mark comment as spam using SpamDetector
                self.logger.info(f"[SPAM] Calling SpamDetector.mark_as_spam({comment_id}, {is_spam})")
                success = self.spam_detector.mark_as_spam(comment_id, is_spam)
                self.logger.info(f"[SPAM] SpamDetector.mark_as_spam returned: {success}")
                
                if not success:
                    self.logger.error(f"[SPAM] SpamDetector.mark_as_spam failed for comment {comment_id}")
                    return {'status': 'error', 'error': 'Failed to mark comment as spam'}
                
                # Get updated metrics
                try:
                    self.logger.info("[SPAM] Calculating updated metrics")
                    metrics = self.spam_detector.calculate_metrics()
                except Exception as e:
                    self.logger.warning(f"[SPAM] Failed to calculate metrics: {e}")
                    metrics = None
                
                self.logger.info(f"[SPAM] Successfully marked comment {comment_id} as {'spam' if is_spam else 'not spam'}")
                return {
                    'status': 'success',
                    'comment_id': comment_id,
                    'is_spam': is_spam,
                    'metrics': metrics
                }
                
            except Exception as e:
                self.logger.error(f"Error in SpamDetector operations: {e}")
                raise
            
        except Exception as e:
            log_error(self.logger, e, f"Failed to mark comment {comment_id} as spam")
            if conn:
                conn.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            if conn:
                conn.close()

    def mark_comments_as_spam(self, comment_ids, is_spam):
        """Mark multiple comments as spam/not spam in bulk."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First check if all comments exist
            placeholders = ','.join(['?' for _ in comment_ids])
            cursor.execute(f"""
                SELECT comment_id, text FROM comments 
                WHERE comment_id IN ({placeholders})
            """, comment_ids)
            
            existing_comments = cursor.fetchall()
            if len(existing_comments) != len(comment_ids):
                found_ids = {comment[0] for comment in existing_comments}
                missing_ids = [id for id in comment_ids if id not in found_ids]
                return {
                    'status': 'error',
                    'error': f'Comments not found: {", ".join(missing_ids)}'
                }
            
            # Mark each comment and add to training data
            processed = 0
            for comment_id, text in existing_comments:
                try:
                    self.spam_detector.mark_as_spam(comment_id, is_spam)
                    processed += 1
                except Exception as e:
                    log_error(self.logger, e, f"Failed to mark comment {comment_id}")
            
            # Check if we should retrain
            cursor.execute("""
                SELECT COUNT(*) FROM spam_training 
                WHERE trained_at > (
                    SELECT COALESCE(MAX(created_at), '1970-01-01')
                    FROM model_versions
                    WHERE model_type = 'spam'
                )
            """)
            
            new_samples = cursor.fetchone()[0]
            conn.close()
            
            # Retrain if we have enough new samples
            retrain_status = None
            if new_samples >= 10:
                metrics = self.spam_detector.train()
                retrain_status = 'Model retrained with new samples'
                
            return {
                'status': 'success',
                'processed': processed,
                'total': len(comment_ids),
                'retrain_status': retrain_status
            }

            
        except Exception as e:
            log_error(self.logger, e, f"Failed to mark comments as spam")
            return {'status': 'error', 'error': str(e)}

    def get_toxicity_scores(self, text):
        """Perform detailed toxicity analysis."""
        try:
            scores = self.toxicity_pipeline(text)[0]
            
            # Get the highest scoring category
            max_score = max(scores, key=lambda x: x['score'])
            
            # Get all high-scoring categories (>0.3)
            significant_categories = [
                s for s in scores 
                if s['score'] > 0.3
            ]
            
            return {
                'label': max_score['label'],
                'score': max_score['score'],
                'details': {
                    'main_category': max_score['label'],
                    'main_score': max_score['score'],
                    'other_concerns': [
                        {'category': s['label'], 'score': s['score']}
                        for s in significant_categories
                        if s != max_score
                    ]
                }
            }
            
        except Exception as e:
            log_error(self.logger, e, "Error in toxicity analysis")
            return {'label': 'unknown', 'score': 0.0, 'details': {}}

    def get_analysis_summary(self, video_id):
        """Get summary of comment analysis for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get classification counts
        cursor.execute("""
            SELECT classification, COUNT(*) 
            FROM comments 
            WHERE video_id = ? 
            GROUP BY classification
        """, (video_id,))
        classifications = dict(cursor.fetchall())

        # Get moderation action counts
        cursor.execute("""
            SELECT mod_action, COUNT(*) 
            FROM comments 
            WHERE video_id = ? AND mod_action IS NOT NULL
            GROUP BY mod_action
        """, (video_id,))
        mod_actions = dict(cursor.fetchall())

        # Get sentiment distribution
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN emotional_score >= 0.3 THEN 1 END) as positive,
                COUNT(CASE WHEN emotional_score BETWEEN -0.3 AND 0.3 THEN 1 END) as neutral,
                COUNT(CASE WHEN emotional_score <= -0.3 THEN 1 END) as negative
            FROM comments 
            WHERE video_id = ?
        """, (video_id,))
        sentiment_dist = cursor.fetchone()

        # Get top keywords (simple implementation)
        cursor.execute("SELECT text FROM comments WHERE video_id = ?", (video_id,))
        comments = cursor.fetchall()
        
        words = []
        for comment in comments:
            # Extract words, remove punctuation, convert to lowercase
            words.extend(re.findall(r'\w+', comment[0].lower()))
        
        # Remove common words and get top 10
        common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                       'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
        keywords = [word for word in words if word not in common_words]
        top_keywords = Counter(keywords).most_common(10)

        conn.close()

        return {
            'classifications': classifications,
            'mod_actions': mod_actions,
            'sentiment_distribution': {
                'positive': sentiment_dist[0],
                'neutral': sentiment_dist[1],
                'negative': sentiment_dist[2]
            },
            'top_keywords': dict(top_keywords)
        }

def main():
    """Test the comment analyzer."""
    analyzer = CommentAnalyzer()
    video_id = input("Enter YouTube video ID to analyze: ")
    analyzer.analyze_comments(video_id)

if __name__ == '__main__':
    main()
