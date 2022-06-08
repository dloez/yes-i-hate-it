"""Process tweets and generate bag of worlds"""
import re
import sys
import random
import logging
import nltk
import stanza
import multiprocessing as mp
from gensim.models import Word2Vec, KeyedVectors
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import CLUSTER_LOG_FILE
from yes_i_hate_it.config import WORD2VEC_MODEL_PATH, WORD2VEC_WV_PATH


def get_tweets(session, amount):
    """Return n amount of not processed tweets from database"""
    # pylint: disable = singleton-comparison
    tweets = session.query(Tweet).filter(Tweet.processed==True).limit(amount).all()
    # for tweet in tweets:
    #     tweet.processed = True
    # session.add_all(tweets)
    # session.commit()
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


def create_clusters(words):
    """Create clusters"""
    word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH), mmap='r')
    # c1 = {'words': [], 'mean_similarity': 0}
    clusters = {} #{'c1': c1}
    clustered_words = {} #{'porterazos': 'c1'}
    new_cluster = 1
    print('start for')
    for word in words:
        word_similarities = max_normalized_sim(word_vectors, word)
        best_similarity = 0
        for sim in word_similarities:
            if sim[0] in clustered_words:
                similarity = sim[1] * clusters[clustered_words[sim[0]]]['mean_similarity']
            else:
                similarity = sim[1]
            if similarity > best_similarity:
                best_similarity = similarity
                word_to_append = sim[0]

        # if word in clustered_words:
        #     word_cluster = clustered_words[word]
        #     if word_to_append in clustered_words:
        #         if word_cluster == clustered_words[word_to_append]:
        #             continue

        #     clusters[word_cluster]['words'].remove(word)
        #     clusters[word_cluster]['mean_similarity'] = calculate_mean_similarity(word_vectors, clusters[word_cluster]['words'])

        if word_to_append in clustered_words:
            cluster = clustered_words[word_to_append]
            clustered_words[word] = cluster
            if word not in clusters[cluster]['words']:
                clusters[cluster]['words'].append(word)
            clusters[cluster]['mean_similarity'] = calculate_mean_similarity(word_vectors, clusters[cluster]['words'])
        else:
            cluster_name = f'c{new_cluster}'
            clustered_words[word] = cluster_name
            clustered_words[word_to_append] = cluster_name
            cluster = {'words': [word, word_to_append], 'mean_similarity': best_similarity}
            clusters[cluster_name] = cluster
            new_cluster += 1

        print(
            f"Words: {len(clustered_words)}/{len(words)} || Clusters: {len(clusters)} || {proportional_hashtags(len(clustered_words),len(words))}",
            end='\r'
        )
        sys.stdout.flush()
    print('')
    # print(clustered_words)
    # print(clusters)
    represent_data(clusters)


def max_normalized_sim(word_vectors, word):
    """Return normalized vector of maximum similarities"""
    similarities = word_vectors.most_similar(word)
    normalized_sim = []
    for sim in similarities:
        normalized_sim.append((sim[0], (sim[1]+1)/2))

    return normalized_sim


def calculate_mean_similarity(word_vectors, words):
    """Return mean simalrity in clusters"""
    mean = 0
    total_words = len(words)
    for i, word in enumerate(words):
        mean += sum([word_vectors.similarity(word, words[j]) for j in range(i+1, total_words)]) / (total_words - i)
    mean /= total_words

    return mean


def proportional_hashtags(current_words, total_words):
    """Return string with progresson bar"""
    hashtags_amount = round(current_words * 20 / total_words) 
    hashtags = '['
    for i in range(1, 20):
        if i < hashtags_amount:
            hashtags += '#'
        else:
            hashtags += ' '
    hashtags += ']'

    return hashtags


def represent_data(clusters):
    """Create and save plots with cluster data"""
    x = []
    n = []
    means = []
    for i, cluster in enumerate(clusters):
        x.append(i+1)
        print(clusters[cluster]['words'])
        n.append(len(clusters[cluster]['words']))
        means.append(clusters[cluster]['mean_similarity'])

    normalize_n = [float(i)/max(n) for i in n]

    plt.plot(x, normalize_n, 'r', label='n elements')
    plt.plot(x, means, 'b', label='means')
    plt.legend(loc='upper left')
    plt.xlabel('Cluster')
    plt.ylabel('n elements')
    plt.savefig('data/elem_sim.png')
    plt.clf()

    plt.plot(x, n)
    plt.xlabel('Cluster')
    plt.ylabel('n elements')
    plt.savefig('data/elements.png')
    plt.clf()

    plt.plot(x, means)
    plt.xlabel('Cluster')
    plt.ylabel('mean similarity')
    plt.savefig('data/similarity.png')
    plt.clf()


def process_word(actions_queue, words_queue, poisson_queue, clusters):
    """Return if a word should be added to a cluster or create a new one"""
    while True:
        word = words_queue.get()
        action = {'word': word, 'name': ''}
        word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH), mmap='r')

        # first batch of words will trigger a new cluster creation for all of them
        # should this be handled? does it really matter?
        if not clusters:
            action['name'] = NEW_CLUSTER
            actions_queue.put(action)
            poisson_queue.put('finished')
            continue
                
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
        actions_queue.put(action)
        poisson_queue.put('finished')


def chunk_words(words, size):
    """Split words set into small chunks"""
    return (words[pos:pos+size] for pos in range(0, len(words), size))


def generate_clusters(words):
    """Generate group of words called 'clusters'"""
    # instantiate cluster dictionary on a shared memory block
    manager = mp.Manager()
    clusters = manager.dict()

    # instantiate queue for multiprocessing communication
    actions_queue = mp.Queue()
    words_queue = mp.Queue()
    poisson_queue = mp.Queue()

    new_cluster = 0
    size = mp.cpu_count()
    processes_count = mp.cpu_count()

    processes = []
    for i in range(processes_count):
        process = mp.Process(target=process_word, args=(actions_queue, words_queue, poisson_queue, clusters))
        process.start()
        processes.append(process)

    for i in range(EPOCHS):
        logging.warning('Epoch %s', i+1)
        random.shuffle(words)
        current_words = 0
        for x, chunk in enumerate(chunk_words(words, size)):
            current_words += len(chunk)
            for word, process in zip(chunk, processes):
                words_queue.put(word)
            
            returns = []
            for process in processes:
                returns.append(actions_queue.get())
            
            poisson_pills = 0
            while poisson_pills < processes_count:
                poisson_queue.get()
                poisson_pills += 1

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

    tweets = get_tweets(session, 100)
    processed_tweets = []
    for tweet in tweets:
        logging.warning("Processing tweet with ID: %s", tweet.tweet_id)
        processed_tweets.append(process_text(tweet.text))

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
    create_clusters(clean_words)


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
