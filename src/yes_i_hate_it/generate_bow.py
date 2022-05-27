"""Process tweets and generate bag of worlds"""
import re
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH


def get_tweets(session, amount):
    """Return n amount of not processed tweets from database"""
    # pylint: disable = singleton-comparison
    tweets = session.query(Tweet).filter(Tweet.processed==False).limit(amount).all()
    for tweet in tweets:
        tweet.processed = True
    session.add_all(tweets)
    session.commit()
    return tweets


def process_text(text):
    """Normalize and return list of words from text"""
    processed_text = text.lower()
    processed_text = re.sub(r'\d+', '', processed_text)
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    processed_text = [word.strip() for word in processed_text.split()]
    return processed_text


def main():
    """Main function"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    _ = get_tweets(session_maker(), 10)


if __name__ == '__main__':
    main()
