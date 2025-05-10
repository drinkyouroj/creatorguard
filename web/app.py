from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, login_required, logout_user, current_user
import sqlite3
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.auth import (
    login_manager, User, register_user, verify_user, create_password_reset_token, 
    verify_reset_token, reset_password, get_pending_users, get_all_users, 
    update_user_status, admin_required, init_auth_db
)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-this')
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
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user, error = verify_user(username, password)
        
        if user:
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash(error or 'Invalid username or password')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if register_user(username, email, password):
            flash('Registration successful! Please wait for an administrator to activate your account.')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists')
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for('login'))

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset request."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        token = create_password_reset_token(email)
        
        if token:
            # In a production environment, send this via email
            reset_url = url_for('reset_password_with_token', token=token, _external=True)
            flash(f'Password reset link (for development): {reset_url}')
        else:
            flash('Email address not found')
            
    return render_template('reset_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password_with_token(token):
    """Handle password reset with token."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
        elif reset_password(token, password):
            flash('Password has been reset successfully')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired reset link')
            
    return render_template('reset_password.html', token=token)

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Render the admin dashboard."""
    pending_users = get_pending_users()
    all_users = get_all_users()
    return render_template('admin.html', pending_users=pending_users, all_users=all_users)

@app.route('/admin/users/<int:user_id>/activate', methods=['POST'])
@login_required
@admin_required
def activate_user(user_id):
    """Activate a user."""
    update_user_status(user_id, True)
    flash('User activated successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users/<int:user_id>/toggle', methods=['POST'])
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
    return redirect(url_for('admin_dashboard'))

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
