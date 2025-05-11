// Global state
let currentPage = 1;
let currentVideoId = null;
let currentFilters = {
    content: null,  // spam, toxic, questionable, safe
    sentiment: null, // positive, neutral, negative
    search: ''
};
let charts = {
    classification: null,
    sentiment: null,
    spamMetrics: null,
    timeline: null
};

// Constants for classification
const CONTENT_TYPES = ['spam', 'toxic', 'questionable', 'safe'];
const SENTIMENT_TYPES = ['positive', 'neutral', 'negative'];
const CONTENT_COLORS = {
    'spam': '#FF9800',     // Orange
    'toxic': '#F44336',    // Red
    'questionable': '#FFC107', // Amber
    'safe': '#4CAF50'      // Green
};
const SENTIMENT_COLORS = {
    'positive': '#2196F3',  // Blue
    'neutral': '#9E9E9E',   // Gray
    'negative': '#673AB7'   // Deep Purple
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadVideos();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Video selection
    document.getElementById('videoSelect').addEventListener('change', (e) => {
        currentVideoId = e.target.value;
        if (currentVideoId) {
            loadVideoData(currentVideoId);
            loadComments(currentVideoId, 1);
        }
    });

    // Pagination
    document.getElementById('prevPage').addEventListener('click', () => {
        if (currentPage > 1) {
            loadComments(currentVideoId, currentPage - 1);
        }
    });

    document.getElementById('nextPage').addEventListener('click', () => {
        loadComments(currentVideoId, currentPage + 1);
    });

    // Video import
    document.getElementById('importForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const videoId = document.getElementById('newVideoId').value.trim();
        if (!videoId) {
            setImportStatus('Please enter a video ID', 'error');
            return;
        }

        setImportStatus('Importing comments...', 'info');
        try {
            const response = await fetch('/api/videos/import', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `video_id=${encodeURIComponent(videoId)}`
            });

            const data = await response.json();
            if (data.success) {
                setImportStatus(data.message, 'success');
                document.getElementById('newVideoId').value = '';
                loadVideos();  // Refresh video list
            } else {
                setImportStatus(data.error || 'Import failed', 'error');
            }
        } catch (error) {
            setImportStatus('Error importing video', 'error');
            console.error('Import error:', error);
        }
    });
}

// API Calls
async function loadVideos() {
    try {
        const response = await fetch('/api/videos');
        const videos = await response.json();
        
        const select = document.getElementById('videoSelect');
        select.innerHTML = '<option value="">Select a video</option>' +
            videos.map(video => `
                <option value="${video.video_id}">
                    ${video.video_id} (${video.comment_count} comments)
                </option>
            `).join('');
    } catch (error) {
        console.error('Error loading videos:', error);
    }
}

async function loadVideoData(videoId) {
    try {
        const response = await fetch(`/api/insights/${videoId}`);
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error loading video data:', error);
    }
}

async function loadComments(videoId, page) {
    try {
        const response = await fetch(`/api/comments/${videoId}?page=${page}`);
        const data = await response.json();
        updateCommentsTable(data);
        currentPage = page;
    } catch (error) {
        console.error('Error loading comments:', error);
    }
}

// UI Updates
function updateDashboard(data) {
    updateStats(data);
    updateCharts(data);
    updateKeywords(data);
}

function updateStats(data) {
    document.getElementById('totalComments').textContent = data.total_comments;
    document.getElementById('uniqueAuthors').textContent = data.unique_authors;
    document.getElementById('responseRate').textContent = `${data.engagement_metrics.response_rate}%`;
    document.getElementById('avgWords').textContent = data.keyword_analysis.avg_words_per_comment.toFixed(1);
}

function updateCharts(data) {
    // Classification Chart
    if (charts.classification) charts.classification.destroy();
    const classCtx = document.getElementById('classificationChart').getContext('2d');
    charts.classification = new Chart(classCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(data.classification_breakdown),
            datasets: [{
                data: Object.values(data.classification_breakdown),
                backgroundColor: [
                    '#4CAF50', // positive
                    '#F44336', // negative
                    '#9E9E9E', // neutral
                    '#FF9800', // offensive
                    '#795548'  // spam
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Moderation Chart
    if (charts.moderation) charts.moderation.destroy();
    const modCtx = document.getElementById('moderationChart').getContext('2d');
    charts.moderation = new Chart(modCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data.mod_action_breakdown),
            datasets: [{
                data: Object.values(data.mod_action_breakdown),
                backgroundColor: [
                    '#4CAF50', // allow
                    '#FFC107', // flag
                    '#F44336'  // remove
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Timeline Chart
    if (charts.timeline) charts.timeline.destroy();
    const timeCtx = document.getElementById('timelineChart').getContext('2d');
    const timeData = processTimeData(data.time_analysis);
    charts.timeline = new Chart(timeCtx, {
        type: 'line',
        data: {
            labels: timeData.labels,
            datasets: [{
                label: 'Comments',
                data: timeData.data,
                borderColor: '#2196F3',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updateKeywords(data) {
    const keywords = data.keyword_analysis.top_keywords;
    const cloudDiv = document.getElementById('keywordCloud');
    cloudDiv.innerHTML = '';
    
    Object.entries(keywords).forEach(([word, count]) => {
        const size = Math.max(1, Math.min(4, Math.log2(count)));
        const span = document.createElement('span');
        span.textContent = word;
        span.className = `inline-block m-1 px-2 py-1 bg-blue-${Math.round(size * 100)} text-white rounded`;
        span.style.fontSize = `${size * 0.5 + 0.8}rem`;
        cloudDiv.appendChild(span);
    });
}

async function markAsSpam(commentId, isSpam) {
    try {
        // Ensure isSpam is a boolean
        isSpam = Boolean(isSpam);
        console.log(`[SPAM] Marking comment ${commentId} as spam=${isSpam}`);
        const requestData = { is_spam: isSpam };
        console.log('[SPAM] Request data:', requestData);
        
        const response = await fetch(`/api/comments/${commentId}/mark_spam`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        console.log(`[SPAM] Response status: ${response.status}`);
        const data = await response.json();
        console.log('[SPAM] Response data:', data);
        
        if (response.ok) {
            console.log('[SPAM] Request successful');
            showNotification(isSpam ? 'Marked as spam' : 'Marked as not spam', 'success');
            // Refresh comments to show updated status
            loadComments(currentVideoId, currentPage);
            return true;
        } else {
            console.error('[SPAM] Request failed:', data.error);
            showNotification(data.error || 'Failed to mark comment as spam', 'error');
            return false;
        }
    } catch (error) {
        console.error('[SPAM] Error marking comment as spam:', error);
        showNotification('Error marking comment as spam', 'error');
        return false;
    }
}

function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    if (!notification) {
        const div = document.createElement('div');
        div.id = 'notification';
        div.className = 'fixed top-4 right-4 p-4 rounded shadow-lg transition-opacity duration-500';
        document.body.appendChild(div);
    }
    
    const notificationEl = document.getElementById('notification');
    notificationEl.textContent = message;
    notificationEl.className = `fixed top-4 right-4 p-4 rounded shadow-lg transition-opacity duration-500 ${
        type === 'error' ? 'bg-red-500 text-white' :
        type === 'success' ? 'bg-green-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    
    setTimeout(() => {
        notificationEl.style.opacity = '0';
        setTimeout(() => notificationEl.style.display = 'none', 500);
    }, 3000);
}

function updateCommentsTable(data) {
    const tbody = document.getElementById('commentsTable');
    
    // Process comments to extract content and sentiment from combined classification
    const processedComments = data.comments.map(comment => {
        // Parse the combined classification (format: "content:sentiment")
        let contentClass = 'unclassified';
        let sentimentClass = 'neutral';
        
        if (comment.classification && comment.classification.includes(':')) {
            const parts = comment.classification.split(':');
            contentClass = parts[0];
            sentimentClass = parts[1];
        } else if (comment.classification) {
            contentClass = comment.classification;
        }
        
        return {
            ...comment,
            contentClass,
            sentimentClass
        };
    });
    
    // Apply filters
    const filteredComments = processedComments.filter(comment => {
        // Content filter
        if (currentFilters.content && comment.contentClass !== currentFilters.content) {
            return false;
        }
        
        // Sentiment filter
        if (currentFilters.sentiment && comment.sentimentClass !== currentFilters.sentiment) {
            return false;
        }
        
        // Search filter
        if (currentFilters.search && !comment.text.toLowerCase().includes(currentFilters.search.toLowerCase())) {
            return false;
        }
        
        return true;
    });
    
    // Update the table
    tbody.innerHTML = filteredComments.map(comment => `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${comment.author}</td>
            <td class="px-6 py-4 text-sm text-gray-500">${comment.text}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div class="flex flex-col space-y-1">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                        style="background-color: ${CONTENT_COLORS[comment.contentClass] || '#9E9E9E'}25; color: ${CONTENT_COLORS[comment.contentClass] || '#9E9E9E'}">
                        ${comment.contentClass || 'unclassified'}
                    </span>
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                        style="background-color: ${SENTIMENT_COLORS[comment.sentimentClass] || '#9E9E9E'}25; color: ${SENTIMENT_COLORS[comment.sentimentClass] || '#9E9E9E'}">
                        ${comment.sentimentClass || 'neutral'}
                    </span>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div class="flex flex-col space-y-2">
                    <div>
                        <span class="text-xs font-medium">Spam Score:</span>
                        <span class="ml-1 text-xs ${comment.spam_score > 0.5 ? 'text-orange-600' : 'text-green-600'}">
                            ${(comment.spam_score * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div>
                        <span class="text-xs font-medium">Toxicity:</span>
                        <span class="ml-1 text-xs ${comment.toxicity_score > 0.5 ? 'text-red-600' : 'text-green-600'}">
                            ${(comment.toxicity_score * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div class="flex flex-col space-y-2">
                    <button onclick="markAsSpam('${comment.comment_id}', true)" 
                        class="${Boolean(comment.is_spam) ? 'bg-orange-500' : 'bg-gray-200'} text-white px-2 py-1 rounded text-xs">
                        Mark as Spam
                    </button>
                    <button onclick="markAsSpam('${comment.comment_id}', false)" 
                        class="${!Boolean(comment.is_spam) ? 'bg-green-500' : 'bg-gray-200'} text-white px-2 py-1 rounded text-xs">
                        Not Spam
                    </button>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${formatDate(comment.timestamp)}
            </td>
        </tr>
    `).join('');
    
    // Update filter counts
    updateFilterCounts(processedComments);
    // Update pagination
    document.getElementById('pageInfo').textContent = `Page ${data.page} of ${data.total_pages}`;
    document.getElementById('prevPage').disabled = data.page <= 1;
    document.getElementById('nextPage').disabled = data.page >= data.total_pages;
}

// Utility Functions
function getClassificationColor(classification) {
    const colors = {
        'positive': 'bg-green-100 text-green-800',
        'negative': 'bg-red-100 text-red-800',
        'neutral': 'bg-gray-100 text-gray-800',
        'offensive': 'bg-orange-100 text-orange-800',
        'spam': 'bg-yellow-100 text-yellow-800'
    };
    return colors[classification] || 'bg-gray-100 text-gray-800';
}

function getActionColor(action) {
    const colors = {
        'allow': 'bg-green-100 text-green-800',
        'flag': 'bg-yellow-100 text-yellow-800',
        'remove': 'bg-red-100 text-red-800'
    };
    return colors[action] || 'bg-gray-100 text-gray-800';
}

function formatDate(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function processTimeData(timeAnalysis) {
    // Process time data for the timeline chart
    // This is a simplified version - you might want to add more sophisticated time series analysis
    return {
        labels: ['First Comment', 'Most Active', 'Last Comment'],
        data: [0, 1, 0]  // Placeholder data - replace with actual time-series data
    };
}

function setImportStatus(message, type = 'info') {
    const statusEl = document.getElementById('importStatus');
    statusEl.textContent = message;
    statusEl.className = `mt-2 text-sm ${
        type === 'error' ? 'text-red-600' :
        type === 'success' ? 'text-green-600' :
        'text-gray-500'
    }`;
}
