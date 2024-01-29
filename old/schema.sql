/**
 * Holds logic for database persistence.
**/

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agents (
    /* rowid */
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    destroyed_at INTEGER DEFAULT NULL,

    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS messages (
    /* rowid */
    agent_id INTEGER NOT NULL,
    channel TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,

    FOREIGN KEY(agent_id) REFERENCES agents(rowid)
);

CREATE TABLE IF NOT EXISTS steps (
    /* rowid */
    message_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,

    FOREIGN KEY(message_id) REFERENCES messages(rowid)
);

CREATE TABLE IF NOT EXISTS pushes (
    message_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,

    PRIMARY KEY(message_id, agent_id),
    FOREIGN KEY(message_id) REFERENCES messages(rowid),
    FOREIGN KEY(agent_id) REFERENCES agents(rowid)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS logs (
    /* rowid */
    level INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
)