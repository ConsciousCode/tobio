/**
 * Persistent storage for the application.
**/
CREATE TABLE IF NOT EXISTS persistent (
    key TEXT PRIMARY KEY,
    value TEXT -- JSON
) WITHOUT ROWID;

/**
 * Message authors.
**/
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    name TEXT,
    
    UNIQUE(role, name)
);

/**
 * Steps in a message.
**/
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES steps(rowid) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES authors(rowid) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    content TEXT NOT NULL
);

UPDATE steps SET status = 'failed' WHERE status = 'stream';