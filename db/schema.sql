-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    comment_id TEXT UNIQUE NOT NULL,
    parent_id TEXT,
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    classification TEXT,
    mod_action TEXT,
    emotional_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES comments (comment_id)
);

-- Create indices for better query performance
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_id ON comments(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_timestamp ON comments(timestamp);
CREATE INDEX IF NOT EXISTS idx_comments_classification ON comments(classification);
CREATE INDEX IF NOT EXISTS idx_comments_mod_action ON comments(mod_action);
