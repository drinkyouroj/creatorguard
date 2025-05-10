from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from .auth import admin_required, User, update_user_status, get_pending_users, get_all_users

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

# API Routes
@bp.route('/api/videos/import', methods=['POST'])
@login_required
def import_video():
    """Import comments from a YouTube video."""
    video_id = request.form.get('video_id')
    if not video_id:
        return jsonify({'success': False, 'error': 'No video ID provided'})
        
    try:
        # Import video logic here
        return jsonify({
            'success': True,
            'message': f'Successfully imported comments for video {video_id}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/videos')
@login_required
def list_videos():
    """List all imported videos."""
    # Video listing logic here
    return jsonify([])

@bp.route('/api/insights/<video_id>')
@login_required
def get_insights(video_id):
    """Get insights for a specific video."""
    # Insights logic here
    return jsonify({})

@bp.route('/api/comments/<video_id>')
@login_required
def get_comments(video_id):
    """Get comments for a specific video."""
    page = request.args.get('page', 1, type=int)
    # Comments logic here
    return jsonify({
        'comments': [],
        'page': page,
        'total_pages': 1
    })
