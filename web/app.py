from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import login_required, login_user, logout_user, current_user
import sqlite3
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.comment_insights import CommentAnalyzer
from comments.fetch_comments import YouTubeCommentFetcher
from auth import login_manager, init_auth_db, register_user, verify_user

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-this')  # Change in production
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize authentication database
init_auth_db()

@app.route('/')
@login_required
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = verify_user(username, password)
        
        if user:
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if register_user(username, email, password):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        flash('Username or email already exists')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/videos')
@login_required
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
@login_required
def get_insights(video_id):
    """Get insights for a specific video."""
    try:
        analyzer = CommentAnalyzer()
        stats = analyzer.get_comment_stats(video_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comments/<video_id>')
@login_required
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

@app.route('/api/videos/import', methods=['POST'])
@login_required
def import_video():
    """Import comments from a new YouTube video."""
    video_id = request.form.get('video_id')
    if not video_id:
        return jsonify({'error': 'Video ID is required'}), 400
    
    try:
        fetcher = YouTubeCommentFetcher()
        comment_count = fetcher.fetch_comments(video_id)
        return jsonify({
            'success': True,
            'message': f'Successfully imported {comment_count} comments',
            'video_id': video_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
