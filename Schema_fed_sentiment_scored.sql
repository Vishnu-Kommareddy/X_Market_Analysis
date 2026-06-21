CREATE TABLE fed_sentiment_scored (
    id SERIAL PRIMARY KEY,
    tweet_text TEXT NOT NULL,
    polarity REAL,                 -- Sentiment polarity score (-1 to +1)
    sentiment TEXT                 -- 'positive', 'negative', or 'neutral'
);
