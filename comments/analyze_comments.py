import os
import sqlite3
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import Counter
import re

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        """Initialize the comment analyzer."""
        self.db_path = db_path
        self.sia = SentimentIntensityAnalyzer()
        # Load toxicity classifier
        self.toxicity = pipeline('text-classification', 
                               model='unitary/toxic-bert', 
                               return_all_scores=True)

    def analyze_comments(self, video_id):
        """Analyze all unanalyzed comments for a video."""
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

        print(f"🔍 Analyzing {len(comments)} comments for video {video_id}")
        
        for comment_id, text in comments:
            # Get sentiment scores
            sentiment_scores = self.sia.polarity_scores(text)
            compound_score = sentiment_scores['compound']
            
            # Get toxicity scores
            try:
                toxicity_scores = self.toxicity(text)[0]
                toxicity_level = max(toxicity_scores, key=lambda x: x['score'])
            except Exception as e:
                print(f"⚠️ Error in toxicity analysis: {e}")
                toxicity_level = {'label': 'unknown', 'score': 0.0}

            # Determine classification and moderation action
            classification = self._get_classification(compound_score, toxicity_level)
            mod_action = self._get_mod_action(classification, toxicity_level)

            # Update database
            try:
                cursor.execute("""
                    UPDATE comments 
                    SET classification = ?,
                        mod_action = ?,
                        emotional_score = ?
                    WHERE comment_id = ?
                """, (classification, mod_action, compound_score, comment_id))
            except sqlite3.Error as e:
                print(f"⚠️ Database error: {e}")
                continue

        conn.commit()
        
        # Get analysis summary
        summary = self.get_analysis_summary(video_id)
        print("\n📊 Analysis Summary:")
        print(f"Total Comments Analyzed: {len(comments)}")
        print("\nClassifications:")
        for cls, count in summary['classifications'].items():
            print(f"- {cls}: {count}")
        print("\nModeration Actions:")
        for action, count in summary['mod_actions'].items():
            print(f"- {action}: {count}")

        conn.close()
        return summary

    def _get_classification(self, sentiment_score, toxicity_result):
        """Determine comment classification based on sentiment and toxicity."""
        toxicity_score = next((item['score'] for item in toxicity_result 
                             if item['label'] == 'toxic'), 0)
        
        if toxicity_score > 0.7:
            return 'toxic'
        elif toxicity_score > 0.4:
            return 'questionable'
        elif sentiment_score >= 0.3:
            return 'positive'
        elif sentiment_score <= -0.3:
            return 'negative'
        else:
            return 'neutral'

    def _get_mod_action(self, classification, toxicity_result):
        """Determine moderation action based on classification."""
        toxicity_score = next((item['score'] for item in toxicity_result 
                             if item['label'] == 'toxic'), 0)
        
        if classification == 'toxic':
            return 'hide'
        elif classification == 'questionable':
            return 'flag'
        elif toxicity_score > 0.3:
            return 'review'
        else:
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
