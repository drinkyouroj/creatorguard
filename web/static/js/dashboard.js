// Global state
let currentPage = 1;
let currentVideoId = null;
let charts = {
    classification: null,
    moderation: null,
    timeline: null
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadVideos();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    document.getElementById('videoSelect').addEventListener('change', (e) => {
        currentVideoId = e.target.value;
        if (currentVideoId) {
            loadVideoData(currentVideoId);
            loadComments(currentVideoId, 1);
        }
    });

    document.getElementById('prevPage').addEventListener('click', () => {
        if (currentPage > 1) {
            loadComments(currentVideoId, currentPage - 1);
        }
    });

    document.getElementById('nextPage').addEventListener('click', () => {
        loadComments(currentVideoId, currentPage + 1);
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

function updateCommentsTable(data) {
    const tbody = document.getElementById('commentsTable');
    tbody.innerHTML = data.comments.map(comment => `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${comment.author}</td>
            <td class="px-6 py-4 text-sm text-gray-500">${comment.text}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                    ${getClassificationColor(comment.classification)}">
                    ${comment.classification || 'unclassified'}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                    ${getActionColor(comment.mod_action)}">
                    ${comment.mod_action || 'pending'}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${formatDate(comment.timestamp)}
            </td>
        </tr>
    `).join('');

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
