-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Password reset tokens table
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    channel_title TEXT,
    thumbnail_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    comment_id TEXT NOT NULL UNIQUE,
    parent_id TEXT,
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    timestamp DATETIME NOT NULL,
    classification TEXT,
    mod_action TEXT,
    emotional_score REAL,
    toxicity_score REAL,
    sentiment_scores TEXT,
    toxicity_details TEXT,
    spam_score REAL DEFAULT 0,
    is_spam BOOLEAN,
    spam_features TEXT,  -- JSON field storing extracted features
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos (video_id),
    FOREIGN KEY (parent_id) REFERENCES comments (comment_id)
);

-- Spam training data table
CREATE TABLE IF NOT EXISTS spam_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id TEXT NOT NULL,
    text TEXT NOT NULL,
    features TEXT,  -- JSON field storing extracted features
    is_spam BOOLEAN NOT NULL,
    confidence REAL DEFAULT 1.0,
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (comment_id) REFERENCES comments (comment_id)
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,  -- 'spam', 'toxicity', etc.
    version TEXT NOT NULL,
    metrics TEXT,  -- JSON field storing model metrics
    parameters TEXT,  -- JSON field storing model parameters
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    metrics TEXT NOT NULL,  -- JSON containing metrics data
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_type_date ON model_metrics(model_type, created_at);

-- Indices for better query performance
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_classification ON comments(classification);
CREATE INDEX IF NOT EXISTS idx_comments_mod_action ON comments(mod_action);
CREATE INDEX IF NOT EXISTS idx_comments_parent_id ON comments(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_timestamp ON comments(timestamp);
CREATE INDEX IF NOT EXISTS idx_comments_comment_id ON comments(comment_id);

-- Indices for users table
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Indices for password reset tokens
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token ON password_reset_tokens(token);
