"""Process tweets and generate bag of worlds"""
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
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
    processed_text = re.sub(r'(\s)http*.://\w+', r'\1', text) # remove URLs
    processed_text = unidecode(processed_text)
    processed_text = processed_text.lower() # lowercase text
    processed_text = re.sub(r'\d+', '', processed_text) # remove numbers
    processed_text = re.sub(r'[^\w\s]', '', processed_text) # remove non char/spaces

    # remove stop words and trailing white spaces
    stop_words = [*stopwords.words('spanish'), *stopwords.words('english')]
    processed_words = [word.strip() for word in processed_text.split() if word not in stop_words]
    return processed_words


def main():
    """Main function"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    _ = get_tweets(session_maker(), 10)


# download stopwords
nltk.download('stopwords')

if __name__ == '__main__':
    main()
