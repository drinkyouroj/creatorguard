import os
import sqlite3
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import Counter
import re
from ..utils.logger import setup_logger, log_error, log_warning

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        """Initialize the comment analyzer."""
        self.db_path = db_path
        self.logger = setup_logger('comment_analyzer')
        
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize VADER")
            raise
            
        try:
            # Load toxicity classifier
            self.toxicity = pipeline('text-classification', 
                                   model='unitary/toxic-bert', 
                                   return_all_scores=True)
            self.logger.info("Initialized toxic-bert model")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize toxic-bert")
            raise

    def analyze_comments(self, video_id):
        """Analyze all unanalyzed comments for a video."""
        self.logger.info(f"Starting analysis for video {video_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get unanalyzed comments
            cursor.execute("""
                SELECT comment_id, text 
                FROM comments 
                WHERE video_id = ? 
                AND classification IS NULL
            """, (video_id,))
            comments = cursor.fetchall()

            self.logger.info(f"Found {len(comments)} unanalyzed comments")
            
            analysis_results = {
                'total': len(comments),
                'processed': 0,
                'errors': 0,
                'classifications': Counter()
            }
            
            for comment_id, text in comments:
                try:
                    # Get sentiment scores
                    sentiment_scores = self.sia.polarity_scores(text)
                    compound_score = sentiment_scores['compound']
                    
                    # Get toxicity scores with detailed analysis
                    toxicity_result = self._analyze_toxicity(text)
                    
                    # Determine classification and moderation action
                    classification = self._get_classification(compound_score, toxicity_result)
                    mod_action = self._get_mod_action(classification, toxicity_result)
                    
                    # Update database with detailed analysis
                    cursor.execute("""
                        UPDATE comments 
                        SET classification = ?,
                            mod_action = ?,
                            emotional_score = ?,
                            toxicity_score = ?,
                            sentiment_scores = ?,
                            toxicity_details = ?
                        WHERE comment_id = ?
                    """, (
                        classification,
                        mod_action,
                        compound_score,
                        toxicity_result['score'],
                        str(sentiment_scores),
                        str(toxicity_result['details']),
                        comment_id
                    ))
                    
                    analysis_results['processed'] += 1
                    analysis_results['classifications'][classification] += 1
                    
                except Exception as e:
                    analysis_results['errors'] += 1
                    log_error(self.logger, e, f"Failed to analyze comment {comment_id}")
                    continue

            conn.commit()
            
            # Log analysis results
            self.logger.info(f"Analysis complete: {analysis_results['processed']} processed, "
                           f"{analysis_results['errors']} errors")
            for cls, count in analysis_results['classifications'].items():
                self.logger.info(f"{cls}: {count}")

            # Get and return analysis summary
            summary = self.get_analysis_summary(video_id)
            return summary

        except sqlite3.Error as e:
            log_error(self.logger, e, f"Database error during analysis of video {video_id}")
            raise
        except Exception as e:
            log_error(self.logger, e, f"Unexpected error during analysis of video {video_id}")
            raise
        finally:
            conn.close()

    def _analyze_toxicity(self, text):
        """Perform detailed toxicity analysis."""
        try:
            scores = self.toxicity(text)[0]
            
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

    def _get_classification(self, sentiment_score, toxicity_result):
        """Determine comment classification based on sentiment and toxicity."""
        try:
            if toxicity_result['label'] == 'toxic' and toxicity_result['score'] > 0.7:
                return 'toxic'
            elif toxicity_result['label'] == 'toxic' and toxicity_result['score'] > 0.4:
                return 'questionable'
            elif sentiment_score >= 0.3:
                return 'positive'
            elif sentiment_score <= -0.3:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            log_error(self.logger, e, "Error in classification")
            return 'unknown'

    def _get_mod_action(self, classification, toxicity_result):
        """Determine moderation action based on classification."""
        try:
            if classification == 'toxic':
                return 'hide'
            elif classification == 'questionable':
                return 'flag'
            elif toxicity_result['score'] > 0.3:
                return 'review'
            else:
                return None
        except Exception as e:
            log_warning(self.logger, f"Error determining mod action: {e}")
            return None

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
