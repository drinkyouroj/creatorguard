CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,  -- YouTube comment ID
    video_id TEXT NOT NULL,
    author TEXT,
    text TEXT,
    timestamp TEXT,
    classification TEXT,       -- e.g., "supportive", "toxic", etc.
    mod_action TEXT,           -- e.g., "hide", "flag", "respond", "ignore"
    reason TEXT,               -- GPT explanation for mod_action
    suggested_reply TEXT,
    responded BOOLEAN DEFAULT 0,
    emotional_score REAL,      -- optional sentiment value
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
