"""Process tweets and generate bag of worlds"""
import re
from multiprocessing import Process
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer
from sqlalchemy.exc import IntegrityError

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import PROCESS_NUMBER
from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE


# pylint: disable = too-few-public-methods
class Word(BASE):
    """Word database ORM"""
    __tablename__ = 'words'

    id = Column(Integer, primary_key=True)
    text = Column(String, unique=True)


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
    # remove URLs
    text = re.sub(r'(https|ftp|http):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', '', text)
    text = unidecode(text)
    text = text.lower() # lowercase text
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text) # remove non char/spaces

    # remove stop words and trailing white spaces
    stop_words = [*stopwords.words('spanish'), *stopwords.words('english')]
    return [word.strip() for word in text.split() if word not in stop_words]


def store_bow(session, words):
    """Store words on database"""
    for word in set(words):
        new_word = Word(text=word)
        session.add(new_word)

        try:
            session.commit()
        except IntegrityError:
            print(f'Repeated word: {word}')
            session.rollback()


def worker():
    """Get and process tweets and generates bow"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    tweets = get_tweets(session, 10)
    while tweets:
        small_bow = []
        for tweet in tweets:
            small_bow.extend(process_text(tweet.text))
        store_bow(session, small_bow)
        tweets = get_tweets(session, 10)


def main():
    """Main function"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    BASE.metadata.create_all(engine)

    processes = []
    for _ in range(PROCESS_NUMBER):
        processes.append(Process(target=worker))

    for process in processes:
        process.start()

    for process in processes:
        process.join()


# download stopwords
nltk.download('stopwords')

if __name__ == '__main__':
    main()
