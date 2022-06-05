"""Process tweets and generate bag of worlds"""
import re
import sys
import random
import logging
import nltk
import stanza
import collections
from functools import lru_cache
import multiprocessing as mp
from gensim.models import Word2Vec, KeyedVectors
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer
from sqlalchemy.exc import IntegrityError

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import CLUSTER_LOG_FILE
from yes_i_hate_it.config import WORD2VEC_MODEL_PATH, WORD2VEC_WV_PATH


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


def process_word(word, clusters, queue):
    """Return if a word should be added to a cluster or create a new one"""
    action = {'word': word, 'name': ''}
    word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH), mmap='r')

    # first batch of words will trigger a new cluster creation for all of them
    # should this be handled? does it really matter?
    if not clusters:
        action['name'] = NEW_CLUSTER
        queue.put(action)
        return
                
    best_similarity = 0
    best_cluster = ''
    found_in = ''
    for key, cluster_words in clusters.items():
        similarity, is_found = get_cluster_similarity(word_vectors, cluster_words, word)
        if is_found:
            found_in = key 

        if not best_similarity:
            best_similarity = similarity
            best_cluster = key
            continue

        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = key

    if best_similarity >= MIN_SIMILARITY:
        action['data'] = {}
        if found_in:
            action['name'] = SWAP
            action['data']['from_cluster'] = found_in
            action['data']['to_cluster'] = best_cluster
        else:
            action['name'] = ADD
            action['data']['to_cluster'] = best_cluster
    else:
        action['name'] = NEW_CLUSTER
    queue.put(action)


def chunk_words(words, size):
    """Split words set into small chunks"""
    return (words[pos:pos+size] for pos in range(0, len(words), size))
    

def generate_clusters(words):
    """Generate group of words called 'clusters'"""
    # instantiate cluster dictionary on a shared memory block
    manager = mp.Manager()
    clusters = manager.dict()

    # instantiate queue for multiprocessing communication
    queue = mp.Queue()

    new_cluster = 0
    size = mp.cpu_count()

    for i in range(EPOCHS):
        logging.warning('Epoch %s', i+1)
        random.shuffle(words)
        current_words = 0
        for x, chunk in enumerate(chunk_words(words, size)):
            current_words += len(chunk)
            processes = []
            for word in chunk:
                process = mp.Process(target=process_word, args=(word, clusters, queue))
                process.start()
                processes.append(process)
            
            returns = []
            for process in processes:
                returns.append(queue.get())

            for process in processes:
                process.join()
            
            for ret in returns:
                if ret['name'] == NEW_CLUSTER:
                    key = f'C{new_cluster}'
                    clusters[key] = manager.list()
                    clusters[key].append(ret['word'])
                    new_cluster += 1
                elif ret['name'] == ADD:
                    cluster_key = ret['data']['to_cluster']
                    clusters[cluster_key].append(ret['word'])
                elif ret['name'] == SWAP:
                    from_cluster = ret['data']['from_cluster']
                    to_cluster = ret['data']['to_cluster']

                    # delete word from current cluster holder and delete cluster if it has no words
                    clusters[from_cluster].remove(ret['word'])
                    if not clusters[from_cluster]:
                        clusters.pop(from_cluster)

                    # add word to cluster
                    clusters[to_cluster].append(ret['word'])
  
            hashtags_amount = round(current_words * 20 / len(words)) 
            hashtags = '['
            for i in range(1, 20):
                if i < hashtags_amount:
                    hashtags += '#'
                else:
                    hashtags += ' '
            hashtags += ']'
            print(f'Words: {current_words}/{len(words)} || Clusters: {len(clusters)} || {hashtags}', end='\r')
            sys.stdout.flush() 
        print('')
        

@lru_cache
def get_cluster_similarity(word_vectors, cluster_words, word):
    """Return average similary between words """
    similarity = 0
    # create copy of list to avoid modifying list on multiprocessing
    cluster_words = list(cluster_words)
    
    is_found = False
    if word in cluster_words:
        cluster_words.remove(word)
        is_found = True

    if not cluster_words:
        return -1, is_found

    for cluster_word in cluster_words:
        similarity += word_vectors.similarity(word, cluster_word)
    return similarity / len(cluster_words), is_found


def find_word(word, clusters):
    """Find word across all chunks and if found return chunk key"""
    for key, words in clusters.items():
        if word in words:
            return key


def main():
    """Main function"""
    # pylint: disable = no-member
    CLUSTER_LOG_FILE.parents[0].mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(CLUSTER_LOG_FILE),
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
            logging.warning("Processing tweet with ID: %s", tweet.tweet_id)
            processed_tweets.append(process_text(tweet.text))
        tweets = get_tweets(session, 10)
    
    model =  Word2Vec(min_count=1, epochs=1, vector_size=100)
    model.build_vocab(processed_tweets)
    model.save(str(WORD2VEC_MODEL_PATH))

    word_vectors = model.wv
    word_vectors.save(str(WORD2VEC_WV_PATH))
    del model

    # delete repeated words
    clean_words = set()
    for sentence in processed_tweets:
        for word in sentence:
            clean_words.add(word)
    del processed_tweets
    
    clean_words = list(clean_words)
    generate_clusters(clean_words)


# download stopwords
nltk.download('stopwords')
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

EPOCHS = 10
MIN_SIMILARITY = 0.3

# Acctions returned from processes
ADD = 'ADD'
SWAP = 'SWAP'
NEW_CLUSTER = 'NEW_CLUSTER'

if __name__ == '__main__':
    main()
