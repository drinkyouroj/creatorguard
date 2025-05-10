from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.comment_insights import CommentAnalyzer

app = Flask(__name__)

@app.route('/')
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@app.route('/api/videos')
def get_videos():
    """Get list of videos with comments."""
    try:
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT video_id, 
                   COUNT(*) as comment_count,
                   MIN(timestamp) as first_comment,
                   MAX(timestamp) as last_comment
            FROM comments 
            GROUP BY video_id
            ORDER BY last_comment DESC
        """)
        videos = cursor.fetchall()
        
        return jsonify([{
            'video_id': video[0],
            'comment_count': video[1],
            'first_comment': video[2],
            'last_comment': video[3]
        } for video in videos])
        
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/insights/<video_id>')
def get_insights(video_id):
    """Get insights for a specific video."""
    try:
        analyzer = CommentAnalyzer()
        stats = analyzer.get_comment_stats(video_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comments/<video_id>')
def get_comments(video_id):
    """Get paginated comments for a video."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    offset = (page - 1) * per_page
    
    try:
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM comments WHERE video_id = ?", (video_id,))
        total = cursor.fetchone()[0]
        
        # Get paginated comments
        cursor.execute("""
            SELECT id, author, text, timestamp, classification, mod_action
            FROM comments 
            WHERE video_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (video_id, per_page, offset))
        
        comments = [{
            'id': row[0],
            'author': row[1],
            'text': row[2],
            'timestamp': row[3],
            'classification': row[4],
            'mod_action': row[5]
        } for row in cursor.fetchall()]
        
        return jsonify({
            'comments': comments,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
        
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
