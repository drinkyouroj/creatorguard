from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from .auth import admin_required, User, update_user_status, get_pending_users, get_all_users
import sqlite3
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comments.fetch_comments import YouTubeCommentFetcher
from comments.analyze_comments import CommentAnalyzer
from gpt.comment_insights import CommentAnalyzer as InsightAnalyzer

bp = Blueprint('views', __name__)

@bp.route('/')
@login_required
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@bp.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Render the admin dashboard."""
    pending_users = get_pending_users()
    all_users = get_all_users()
    return render_template('admin.html', pending_users=pending_users, all_users=all_users)

@bp.route('/admin/users/<int:user_id>/activate', methods=['POST'])
@login_required
@admin_required
def activate_user(user_id):
    """Activate a user."""
    update_user_status(user_id, True)
    flash('User activated successfully')
    return redirect(url_for('views.admin_dashboard'))

@bp.route('/admin/users/<int:user_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_user_status(user_id):
    """Toggle a user's status."""
    action = request.form.get('action')
    if action == 'toggle_active':
        user = User.get(user_id)
        update_user_status(user_id, not user.is_active)
        flash(f'User {"activated" if not user.is_active else "deactivated"} successfully')
    elif action == 'toggle_admin':
        user = User.get(user_id)
        if user.id != current_user.id:  # Prevent self-demotion
            update_user_status(user_id, user.is_active, not user.is_admin)
            flash(f'User admin status {"granted" if not user.is_admin else "revoked"} successfully')
    return redirect(url_for('views.admin_dashboard'))

@bp.route('/api/videos')
@login_required
def list_videos():
    """Get list of videos with comments."""
    try:
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                v.video_id,
                v.title,
                v.channel_title,
                v.thumbnail_url,
                COUNT(c.id) as comment_count,
                MIN(c.timestamp) as first_comment,
                MAX(c.timestamp) as last_comment,
                COUNT(CASE WHEN c.classification = 'toxic' THEN 1 END) as toxic_count,
                COUNT(CASE WHEN c.classification = 'questionable' THEN 1 END) as questionable_count,
                COUNT(CASE WHEN c.mod_action IS NOT NULL THEN 1 END) as flagged_count
            FROM videos v
            LEFT JOIN comments c ON v.video_id = c.video_id
            GROUP BY v.video_id, v.title, v.channel_title, v.thumbnail_url
            ORDER BY last_comment DESC
        """)
        videos = cursor.fetchall()
        
        return jsonify([{
            'video_id': video[0],
            'title': video[1],
            'channel_title': video[2],
            'thumbnail_url': video[3],
            'comment_count': video[4],
            'first_comment': video[5],
            'last_comment': video[6],
            'toxic_count': video[7],
            'questionable_count': video[8],
            'flagged_count': video[9]
        } for video in videos])
        
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@bp.route('/api/insights/<video_id>')
@login_required
def get_insights(video_id):
    """Get insights for a specific video."""
    try:
        analyzer = CommentAnalyzer()
        stats = analyzer.get_analysis_summary(video_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/comments/<video_id>')
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
            SELECT id, author, text, timestamp, classification, mod_action, emotional_score
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
            'mod_action': row[5],
            'emotional_score': row[6]
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

@bp.route('/api/videos/import', methods=['POST'])
@login_required
def import_video():
    """Import comments from a new YouTube video."""
    video_id = request.form.get('video_id')
    if not video_id:
        return jsonify({'error': 'Video ID is required'}), 400
    
    try:
        # Import comments
        fetcher = YouTubeCommentFetcher()
        comment_count = fetcher.fetch_comments(video_id)
        
        # Analyze comments
        analyzer = CommentAnalyzer()
        analysis = analyzer.analyze_comments(video_id)
        
        return jsonify({
            'success': True,
            'message': f'Successfully imported and analyzed {comment_count} comments',
            'video_id': video_id,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
