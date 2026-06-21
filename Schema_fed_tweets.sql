DROP TABLE IF EXISTS fed_tweets;

CREATE TABLE fed_tweets (
    id SERIAL PRIMARY KEY,
    tweet_id BIGINT UNIQUE,
    author_id BIGINT,
    tweet_text TEXT NOT NULL,
    created_at TIMESTAMP,
    lang TEXT,
    source TEXT,
    like_count INT,
    retweet_count INT,
    sentiment TEXT
);
