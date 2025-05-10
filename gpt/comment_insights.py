import os
import sqlite3
import json
import openai
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo')

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        """
        Initialize the comment analyzer with a database connection.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_comment_stats(self, video_id=None):
        """
        Generate statistical insights about comments.
        
        Args:
            video_id (str, optional): Specific video to analyze. If None, analyze all comments.
        
        Returns:
            dict: Comment statistics
        """
        # Base query
        query = "SELECT classification, mod_action FROM comments"
        params = []
        
        # Add video_id filter if provided
        if video_id:
            query += " WHERE video_id = ?"
            params.append(video_id)
        
        # Execute query
        self.cursor.execute(query, params)
        comments = self.cursor.fetchall()
        
        # Calculate statistics
        stats = {
            'total_comments': len(comments),
            'classification_breakdown': dict(Counter(comment[0] for comment in comments)),
            'mod_action_breakdown': dict(Counter(comment[1] for comment in comments)),
            'sentiment_ratio': {}
        }
        
        # Calculate sentiment ratios
        total = stats['total_comments']
        for category, count in stats['classification_breakdown'].items():
            stats['sentiment_ratio'][category] = round(count / total * 100, 2)
        
        return stats

    def generate_comprehensive_summary(self, video_id=None):
        """
        Use GPT to generate a comprehensive summary of comment insights.
        
        Args:
            video_id (str, optional): Specific video to analyze
        
        Returns:
            str: Detailed GPT-generated summary
        """
        # Get comment stats
        stats = self.get_comment_stats(video_id)
        
        # Prepare prompt for GPT
        prompt = f"""Analyze the following YouTube comment statistics and provide a comprehensive, insightful summary:

Comment Statistics:
- Total Comments: {stats['total_comments']}
- Classification Breakdown: {json.dumps(stats['classification_breakdown'])}
- Sentiment Ratios: {json.dumps(stats['sentiment_ratio'])}
- Moderation Action Breakdown: {json.dumps(stats['mod_action_breakdown'])}

Please provide a detailed analysis that includes:
1. Overall sentiment of the comments
2. Key insights about audience engagement
3. Potential areas of improvement or concern
4. Recommendations for content strategy

Your response should be professional, constructive, and actionable."""

        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert content and audience insights analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def save_insights(self, insights, filename='comment_insights.md'):
        """
        Save insights to a markdown file.
        
        Args:
            insights (str): Insights text to save
            filename (str): Filename to save insights
        """
        insights_dir = 'insights'
        os.makedirs(insights_dir, exist_ok=True)
        
        filepath = os.path.join(insights_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Comment Insights\n\n")
            f.write(insights)
        
        print(f"✅ Insights saved to {filepath}")

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    # Example usage
    analyzer = CommentAnalyzer()
    
    try:
        # Get stats for all comments
        all_stats = analyzer.get_comment_stats()
        print("Overall Comment Statistics:")
        print(json.dumps(all_stats, indent=2))
        
        # Generate comprehensive summary
        summary = analyzer.generate_comprehensive_summary()
        
        # Save insights to file
        analyzer.save_insights(summary)
    
    except Exception as e:
        print(f"Error analyzing comments: {e}")
    
    finally:
        analyzer.close()

if __name__ == '__main__':
    main()
