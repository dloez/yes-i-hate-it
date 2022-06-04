"""Process tweets and generate bag of worlds"""
import re
import logging
import nltk
import stanza
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer
from sqlalchemy.exc import IntegrityError

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import BOW_LOG_FILE


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
    text = text.replace('_', '') # remove _

    # words that have no meanings
    stop_words = [*stopwords.words('spanish'), *stopwords.words('english')]

    doc = nlp(text)
    words = []
    for sentence in doc.sentences:
        for word_data in sentence.words:
            lemma = word_data.lemma
            if lemma not in stop_words:
                words.append(unidecode(lemma))
    return words


def store_bow(session, words):
    """Store words on database"""
    for word in set(words):
        new_word = Word(text=word)
        session.add(new_word)

        try:
            session.commit()
        except IntegrityError:
            logging.info("%s repeated, skipping...", word)
            session.rollback()


def delete_words():
    """Delete words table to re-generate bow"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    Word.__table__.drop(engine)


def executor():
    """Get and process tweets and generates bow"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    tweets = get_tweets(session, 10)
    processed_tweets = []
    while tweets:
        for tweet in tweets:
            logging.info("Processing tweet with ID: %s", tweet.tweet_id)
            processed_tweets.append(process_text(tweet.text))
        tweets = get_tweets(session, 10)


def main():
    """Main function"""
    # pylint: disable = no-member
    BOW_LOG_FILE.parents[0].mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(BOW_LOG_FILE),
            logging.StreamHandler()
        ]
    )

    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    BASE.metadata.create_all(engine)
    executor()


# download stopwords
nltk.download('stopwords')
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

if __name__ == '__main__':
    main()
