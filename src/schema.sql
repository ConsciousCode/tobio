/**
 * Persistent storage for the application.
**/
CREATE TABLE IF NOT EXISTS persistent (
    key TEXT PRIMARY KEY,
    value TEXT -- JSON
) WITHOUT ROWID;

/**
 * Messages in a conversation.
**/
CREATE TABLE IF NOT EXISTS messages (
    role TEXT NOT NULL,
    name TEXT,
    created_at REAL NOT NULL
);

/**
 * Steps in a message.
**/
CREATE TABLE IF NOT EXISTS steps (
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    content TEXT
);