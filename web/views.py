from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from .auth import admin_required, User, update_user_status, get_pending_users, get_all_users
import sqlite3
import os
import sys
import json
from comments.spam_detector import SpamDetector
from comments.analyze_comments import CommentAnalyzer
from comments.fetch_comments import YouTubeCommentFetcher
from utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                COUNT(CASE WHEN c.is_spam = 1 THEN 1 END) as spam_count
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
            'spam_count': video[9]
        } for video in videos])
        
    except Exception as e:
        logger.error(f"Failed to list videos: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@bp.route('/api/videos/<video_id>/comments')
@login_required
def get_video_comments(video_id):
    """Get comments for a specific video."""
    try:
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                c.comment_id,
                c.text,
                c.author,
                c.timestamp,
                c.classification,
                c.toxicity_score,
                c.is_spam,
                c.spam_score,
                c.spam_features
            FROM comments c
            WHERE c.video_id = ?
            ORDER BY c.timestamp DESC
        """, (video_id,))
        
        comments = cursor.fetchall()
        return jsonify([{
            'comment_id': comment[0],
            'text': comment[1],
            'author': comment[2],
            'timestamp': comment[3],
            'classification': comment[4],
            'toxicity_score': comment[5],
            'is_spam': bool(comment[6]),
            'spam_score': comment[7],
            'spam_features': json.loads(comment[8]) if comment[8] else None
        } for comment in comments])
        
    except Exception as e:
        logger.error(f"Failed to get video comments: {str(e)}")
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
        logger.error(f"Failed to get insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/metrics/spam', methods=['GET'])
@login_required
def get_spam_metrics():
    """Get spam detection metrics and trends."""
    try:
        logger.info("[METRICS] Getting spam detection metrics")
        analyzer = CommentAnalyzer()
        metrics = analyzer.spam_detector.calculate_metrics()
        
        if metrics is None:
            logger.warning("[METRICS] No metrics available")
            return jsonify({
                'accuracy': None,
                'top_features': {},
                'total_samples': 0,
                'spam_samples': 0,
                'ham_samples': 0,
                'model_status': 'uninitialized'
            })
            
        logger.info(f"[METRICS] Got metrics: {metrics}")
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"[METRICS] Failed to get spam metrics: {str(e)}")
        return jsonify({
            'error': str(e),
            'accuracy': None,
            'top_features': {},
            'total_samples': 0,
            'spam_samples': 0,
            'ham_samples': 0,
            'model_status': 'error'
        })

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
            SELECT id, comment_id, author, text, timestamp, classification, mod_action, 
                   emotional_score, is_spam, spam_score
            FROM comments 
            WHERE video_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (video_id, per_page, offset))
        
        comments = [{
            'id': row[0],
            'comment_id': row[1],
            'author': row[2],
            'text': row[3],
            'timestamp': row[4],
            'classification': row[5],
            'mod_action': row[6],
            'emotional_score': row[7],
            'is_spam': bool(row[8]) if row[8] is not None else None,
            'spam_score': float(row[9]) if row[9] is not None else None
        } for row in cursor.fetchall()]
        
        return jsonify({
            'comments': comments,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
        
    except sqlite3.Error as e:
        logger.error(f"Failed to get comments: {str(e)}")
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
        logger.error(f"Failed to import video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/comments/<comment_id>/mark_spam', methods=['POST'])
@login_required
def mark_comment_spam(comment_id):
    """Mark a comment as spam or not spam."""
    try:
        logger.info(f"[SPAM] Received request to mark comment {comment_id} as spam")
        data = request.get_json()
        if data is None:
            logger.error("[SPAM] No JSON data received in request")
            return jsonify({'error': 'No JSON data received'}), 400
            
        is_spam = data.get('is_spam')
        if is_spam is None:
            logger.error("[SPAM] is_spam field missing in request")
            return jsonify({'error': 'is_spam field is required'}), 400
            
        # Convert is_spam to boolean
        try:
            is_spam = bool(is_spam)
        except (ValueError, TypeError):
            logger.error(f"[SPAM] Invalid is_spam value: {is_spam}")
            return jsonify({'error': 'is_spam must be a boolean value'}), 400
            
        logger.info(f"[SPAM] Request data: {data}")
        logger.info(f"[SPAM] Marking comment {comment_id} as spam={is_spam}")
        
        # Verify comment exists
        conn = sqlite3.connect('creatorguard.db')
        cursor = conn.cursor()
        cursor.execute("SELECT comment_id FROM comments WHERE comment_id = ?", (comment_id,))
        result = cursor.fetchone()
        if not result:
            logger.error(f"[SPAM] Comment {comment_id} not found in database")
            return jsonify({'error': f'Comment {comment_id} not found'}), 404
        conn.close()
        
        analyzer = CommentAnalyzer()
        result = analyzer.mark_comment_as_spam(comment_id, is_spam)
        logger.info(f"[SPAM] mark_comment_as_spam returned: {result}")
        
        if result['status'] == 'error':
            logger.error(f"[SPAM] Failed to mark comment as spam: {result['error']}")
            return jsonify({'error': result['error']}), 500
            
        # Return success response with status
        logger.info(f"[SPAM] Successfully marked comment {comment_id} as spam={is_spam}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[SPAM] Failed to mark comment as spam: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/api/comments/mark_spam_bulk', methods=['POST'])
@login_required
def mark_comments_spam_bulk():
    """Mark multiple comments as spam or not spam."""
    try:
        data = request.get_json()
        comment_ids = data.get('comment_ids', [])
        is_spam = data.get('is_spam', True)
        
        if not comment_ids:
            return jsonify({'error': 'No comments specified'}), 400
        
        analyzer = CommentAnalyzer()
        result = analyzer.mark_comments_as_spam(comment_ids, is_spam)
        
        if result['status'] == 'error':
            return jsonify({'error': result['error']}), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to mark comments as spam: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/metrics/spam', methods=['GET'])
@login_required
def get_spam_metrics():
    """Get spam detection metrics and trends."""
    try:
        days = request.args.get('days', default=30, type=int)
        if days <= 0 or days > 365:
            return jsonify({'error': 'Days must be between 1 and 365'}), 400
            
        detector = SpamDetector()
        
        # Get current metrics
        current_metrics = detector.calculate_metrics()
        if current_metrics is None:
            logger.warning("No metrics data available yet - model may need training")
            return jsonify({
                'error': 'No metrics available yet. Try marking some comments as spam to train the model.',
                'needs_training': True
            }), 404
        
        # Get historical metrics
        metrics_history = detector.get_metrics_history(days)
        
        # Get spam trends
        trends = detector.get_spam_trends(days)
        
        return jsonify({
            'current': current_metrics,
            'history': metrics_history,
            'trends': trends
        })
        
    except Exception as e:
        logger.error(f"Failed to get spam metrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
