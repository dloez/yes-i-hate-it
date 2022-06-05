"""Process tweets and generate bag of worlds"""
import re
import logging
import nltk
import stanza
import collections
from gensim.models import Word2Vec
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


def generate_chunks(sentences, keyed_vectors):
    """Generate group of words called 'chunks'"""
    chunks = {}
    new_chunk = 0
    for i in range(EPOCHS):
        logging.info('Epoch %s', i)
        for sentence in sentences:
            for word in sentence:
                if not chunks:
                    key = f'chunk_{new_chunk}'
                    chunks[key] = []
                    chunks[key].append(word)
                    new_chunk += 1
                    continue
                
                found_in = find_word(word, chunks)
                if found_in:
                    chunks[found_in].remove(word)
                    if not chunks[found_in]:
                        chunks.pop(found_in)
                
                best_similarity = 0
                best_chunk = ''
                for key, words in chunks.items():
                    similarity = get_chunk_similarity(keyed_vectors, words, word)

                    if not best_similarity:
                        best_similarity = similarity
                        best_chunk = key
                        continue

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_chunk = key

                if best_similarity >= MIN_SIMILARITY:
                    if word not in chunks[best_chunk]:
                        chunks[best_chunk].append(word)
                else:
                    key = f'chunk_{new_chunk}'
                    chunks[key] = []
                    chunks[key].append(word)
                    new_chunk += 1
    print(chunks)
    print(len(chunks))


def get_chunk_similarity(keyed_vectors, chunk, word):
    """Return average similary between words """
    similarity = 0
    for chunk_word in chunk:
        similarity += keyed_vectors.similarity(word, chunk_word)
    return similarity / len(chunk)


def find_word(word, chunks):
    """Find word across all chunks and if found return chunk key"""
    for key, words in chunks.items():
        if word in words:
            return key


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
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    tweets = get_tweets(session, 10)
    processed_tweets = []
    while tweets:
        for tweet in tweets:
            logging.info("Processing tweet with ID: %s", tweet.tweet_id)
            processed_tweets.append(process_text(tweet.text))
        tweets = get_tweets(session, 10)
    
    model =  Word2Vec(min_count=1, epochs=1, vector_size=100)
    model.build_vocab(processed_tweets)

    keyed_vectors = model.wv
    del model
    generate_chunks(processed_tweets, keyed_vectors)


# download stopwords
nltk.download('stopwords')
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

EPOCHS = 50
MIN_SIMILARITY = 0.3

if __name__ == '__main__':
    main()
