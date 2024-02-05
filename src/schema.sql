/**
 * Persistent storage for the application
**/
CREATE TABLE persistent (
    key TEXT PRIMARY KEY,
    value TEXT -- JSON
);

/**
 * Message authors, including agents.
**/
CREATE TABLE authors (
    guid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    data TEXT --JSON
);

/**
 * Messages in a conversation
**/
CREATE TABLE messages (
    guid TEXT PRIMARY KEY,
    parent_guid TEXT,
    user_guid TEXT NOT NULL,
    created_at INT NOT NULL,
    data TEXT, -- JSON
    
    FOREIGN KEY (parent_guid) REFERENCES messages(guid),
    FOREIGN KEY (user_guid) REFERENCES authors(guid)
);