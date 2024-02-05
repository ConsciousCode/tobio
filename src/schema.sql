/**
 * Persistent storage for the application
**/
CREATE TABLE IF NOT EXISTS persistent (
    key TEXT PRIMARY KEY,
    value TEXT -- JSON
);

/**
 * Messages in a conversation
**/
CREATE TABLE IF NOT EXISTS messages (
    role TEXT NOT NULL,
    name TEXT,
    created_at REAL NOT NULL,
    content TEXT
);