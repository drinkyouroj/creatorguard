import os
import sqlite3
import json
from collections import Counter
from datetime import datetime
import re
from typing import Dict, List, Tuple
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
try:
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        http_client=httpx.Client()  # Explicitly create an httpx client
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# Use a valid OpenAI model
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

class CommentAnalyzer:
    def __init__(self, db_path='creatorguard.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_comment_stats(self, video_id=None):
        """Get comprehensive statistics about comments."""
        query = """
            SELECT 
                c.classification, 
                c.mod_action, 
                c.text,
                c.author,
                c.timestamp,
                c.likes,
                c.reply_count,
                c.parent_id
            FROM comments c
        """
        params = []
        
        if video_id:
            query += " WHERE c.video_id = ?"
            params.append(video_id)
        
        try:
            self.cursor.execute(query, params)
            comments = self.cursor.fetchall()
            
            print(f"\n📊 Analysis for video {video_id if video_id else 'all videos'}:")
            print(f"💬 Total comments found: {len(comments)}")
            
            # Basic statistics
            stats = {
                "total_comments": len(comments),
                "unique_authors": len(set(comment[3] for comment in comments)),
                "classification_breakdown": dict(Counter(comment[0] for comment in comments if comment[0])),
                "mod_action_breakdown": dict(Counter(comment[1] for comment in comments if comment[1])),
                "engagement_metrics": self._calculate_engagement_metrics(comments),
                "time_analysis": self._analyze_time_patterns(comments),
                "top_commenters": self._get_top_commenters(comments),
                "keyword_analysis": self._analyze_keywords(comments),
                "response_patterns": self._analyze_response_patterns(comments),
                "sample_comments": []
            }
            
            # Add sample comments
            print("\n📝 Sample Comments:")
            for comment in comments[:5]:
                comment_info = {
                    "text": comment[2][:100] + "..." if len(comment[2]) > 100 else comment[2],
                    "author": comment[3],
                    "classification": comment[0],
                    "mod_action": comment[1],
                    "timestamp": comment[4],
                    "likes": comment[5],
                    "replies": comment[6]
                }
                stats["sample_comments"].append(comment_info)
                print(f"Classification: {comment[0]}, Mod Action: {comment[1]}, Text: {comment[2][:50]}...")
            
            return stats
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {}

    def _calculate_engagement_metrics(self, comments: List[Tuple]) -> Dict:
        """Calculate engagement metrics from comments."""
        total_likes = sum(comment[5] or 0 for comment in comments)
        total_replies = sum(comment[6] or 0 for comment in comments)
        
        return {
            "avg_likes_per_comment": round(total_likes / len(comments), 2) if comments else 0,
            "avg_replies_per_comment": round(total_replies / len(comments), 2) if comments else 0,
            "total_likes": total_likes,
            "total_replies": total_replies,
            "reply_ratio": round(len([c for c in comments if c[7]]) / len(comments) * 100, 2) if comments else 0
        }

    def _analyze_time_patterns(self, comments: List[Tuple]) -> Dict:
        """Analyze posting time patterns."""
        if not comments:
            return {}
            
        times = []
        for comment in comments:
            try:
                timestamp = datetime.fromisoformat(comment[4].replace('Z', '+00:00'))
                times.append(timestamp)
            except (ValueError, AttributeError):
                continue
        
        if not times:
            return {}
            
        times.sort()
        time_diffs = [(times[i+1] - times[i]).total_seconds() / 3600 for i in range(len(times)-1)]
        
        return {
            "first_comment": times[0].isoformat() if times else None,
            "last_comment": times[-1].isoformat() if times else None,
            "avg_hours_between_comments": round(sum(time_diffs) / len(time_diffs), 2) if time_diffs else 0,
            "most_active_hour": max(Counter(t.hour for t in times).items(), key=lambda x: x[1])[0] if times else None
        }

    def _get_top_commenters(self, comments: List[Tuple]) -> Dict:
        """Get statistics about top commenters."""
        author_stats = {}
        for comment in comments:
            author = comment[3]
            if author not in author_stats:
                author_stats[author] = {
                    "comment_count": 0,
                    "total_likes": 0,
                    "total_replies": 0
                }
            author_stats[author]["comment_count"] += 1
            author_stats[author]["total_likes"] += comment[5] or 0
            author_stats[author]["total_replies"] += comment[6] or 0
        
        # Sort by comment count and get top 5
        top_authors = sorted(
            author_stats.items(),
            key=lambda x: (x[1]["comment_count"], x[1]["total_likes"]),
            reverse=True
        )[:5]
        
        return {author: stats for author, stats in top_authors}

    def _analyze_keywords(self, comments: List[Tuple]) -> Dict:
        """Analyze frequent keywords and phrases."""
        # Combine all comment text
        text = " ".join(comment[2].lower() for comment in comments)
        
        # Remove special characters and split into words
        words = re.findall(r'\w+', text)
        
        # Remove common stop words (simplified list)
        stop_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get word frequencies
        word_freq = Counter(words).most_common(10)
        
        return {
            "top_keywords": dict(word_freq),
            "total_word_count": len(words),
            "avg_words_per_comment": round(len(words) / len(comments), 2) if comments else 0
        }

    def _analyze_response_patterns(self, comments: List[Tuple]) -> Dict:
        """Analyze patterns in comment responses."""
        response_times = []
        thread_depths = Counter()
        
        for comment in comments:
            if comment[7]:  # If it's a reply
                thread_depths[comment[7]] += 1
        
        return {
            "total_threads": len(thread_depths),
            "max_thread_depth": max(thread_depths.values()) if thread_depths else 0,
            "avg_replies_per_thread": round(sum(thread_depths.values()) / len(thread_depths), 2) if thread_depths else 0
        }

    def generate_summary(self, stats):
        """Generate a comprehensive analysis summary."""
        summary = f"""# Comment Analysis Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total Comments: {stats['total_comments']}
- Unique Commenters: {stats['unique_authors']}

## Classification Breakdown
{self._format_dict(stats.get('classification_breakdown', {}))}

## Moderation Actions
{self._format_dict(stats.get('mod_action_breakdown', {}))}

## Engagement Metrics
{self._format_dict(stats.get('engagement_metrics', {}))}

## Time Analysis
{self._format_dict(stats.get('time_analysis', {}))}

## Top Commenters
{self._format_dict(stats.get('top_commenters', {}))}

## Keyword Analysis
{self._format_dict(stats.get('keyword_analysis', {}))}

## Response Patterns
{self._format_dict(stats.get('response_patterns', {}))}

## Sample Comments
"""
        for comment in stats.get('sample_comments', []):
            summary += f"""
- Author: {comment['author']}
  Classification: {comment['classification']}
  Action: {comment['mod_action']}
  Likes: {comment['likes']}
  Replies: {comment['replies']}
  Text: "{comment['text']}"
"""
        return summary

    def _format_dict(self, d):
        """Format dictionary items for display."""
        if not d:
            return "No data available"
        if isinstance(d, dict):
            return "\n".join([f"- {k}: {v}" for k, v in d.items()])
        return str(d)

    def save_insights(self, video_id, summary):
        """Save insights to a markdown file."""
        os.makedirs('insights', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'insights/comment_insights_{video_id}_{timestamp}.md'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n✅ Insights saved to {filename}")

    def generate_comprehensive_summary(self, stats):
        """Generate a comprehensive summary of comment insights using GPT."""
        if client is None:
            return "Error: OpenAI client could not be initialized."

        try:
            prompt = f"""Analyze the following YouTube comment statistics and provide a comprehensive, insightful summary:

Total Comments: {stats['total_comments']}

Classification Breakdown:
{json.dumps(stats['classification_breakdown'], indent=2)}

Moderation Action Breakdown:
{json.dumps(stats['mod_action_breakdown'], indent=2)}

Engagement Metrics:
{json.dumps(stats['engagement_metrics'], indent=2)}

Time Analysis:
{json.dumps(stats['time_analysis'], indent=2)}

Top Commenters:
{json.dumps(stats['top_commenters'], indent=2)}

Keyword Analysis:
{json.dumps(stats['keyword_analysis'], indent=2)}

Response Patterns:
{json.dumps(stats['response_patterns'], indent=2)}

Provide insights into audience engagement, potential content trends, and recommendations for content creators."""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful content analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.choices[0].message.content
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Unable to generate summary: {str(e)}"

def main():
    try:
        video_id = input("🎥 Enter YouTube video ID: ").strip()
        analyzer = CommentAnalyzer()
        
        # Get and display statistics
        stats = analyzer.get_comment_stats(video_id)
        print("\nVideo Comment Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Generate and display summary
        summary = analyzer.generate_summary(stats)
        print("\n--- Comprehensive Summary ---")
        print(summary)
        
        # Generate comprehensive summary using GPT
        gpt_summary = analyzer.generate_comprehensive_summary(stats)
        print("\n--- GPT Summary ---")
        print(gpt_summary)
        
        # Save insights
        analyzer.save_insights(video_id, summary)
        
    except Exception as e:
        print(f"❌ Error analyzing comments: {e}")

if __name__ == '__main__':
    main()
