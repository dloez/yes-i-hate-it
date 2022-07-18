"""Process tweets and generate bag of worlds"""
import re
import sys
import logging
import multiprocessing as mp
import nltk
import stanza
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import CLUSTER_LOG_FILE
from yes_i_hate_it.config import WORD2VEC_MODEL_PATH, WORD2VEC_WV_PATH 
from yes_i_hate_it.config import KMEANS_DATA_PATH, KMEANS_GRAPH_PATH


class Word(BASE):
    """Word base clase to interact with database"""
    __tablename__ = 'words'

    word = Column(String, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), unique=False)
    cluster = relationship('Cluster', back_populates='words')


class Cluster(BASE):
    """Cluster base clase to interact with databse"""
    __tablename__ = 'clusters'

    id = Column(Integer, primary_key=True)
    words = relationship('Word', back_populates='cluster')


def get_tweets(session, amount):
    """Return n amount of not processed tweets from database""" 
    # pylint: disable = singleton-comparison
    tweets = session.query(Tweet).filter(Tweet.processed==False).limit(amount).all()
    tweets_text = []
    for tweet in tweets:
        tweet.processed = True
        tweets_text.append(tweet.text)
    session.add_all(tweets)
    session.commit()
    return tweets_text


def process_text(text, nlp=None):
    """Normalize and return list of words from text"""
    if not nlp:
        nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

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


def save_clusters(clusters, session):
    """Save clusters into database"""
    for key, words in clusters.items():
        word_storage = []
        new_cluster = Cluster(id=key)
        for word in words:
            new_word = Word(word=word, cluster=new_cluster)
            word_storage.append(new_word)
        session.add_all(word_storage)
        session.add(new_cluster)
        session.commit()


def vectorize(keyed_vectors):
    """Generate vectors from copus (tweets) using word2vec model"""
    features = []
    for word in keyed_vectors.key_to_index:
        word_vector = keyed_vectors[word]
        features.append(word_vector)
    return features


def kmeans_clusters(vectors, n_clusters, show_output):
    """Generate clusters and print Silhouette metrics using Kmeans"""
    km = KMeans(n_clusters).fit(vectors)
    print(f"For n_clusters = {n_clusters}")
    print(f"Silhouette coefficient: {silhouette_score(vectors, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if show_output:
        sample_silhouette_values = silhouette_samples(vectors, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km


def process_tweets(tweets_queue, data_queue):
    nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
    
    while True:
        data = []
        tweets = tweets_queue.get()
        for tweet in tweets:
            data.append(process_text(tweet, nlp))
        data_queue.put(data)


def chunk_tweets(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def compose_words_clusters(keyed_vectors, kmeans, n_clusters):
    """Converts kmeans clusters to clusters of words"""
    km_labels = kmeans.labels_
    clusters = {}

    for i, cluster_id in enumerate(km_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(keyed_vectors.index_to_key[i])
    return clusters


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

    corpus = []
    
    chunk_size = 1000
    tweets_queue = mp.Queue()
    data_queue = mp.Queue()
    processes_count = 4
    processes = []
     
    for count in range(processes_count):
        process = mp.Process(target=process_tweets, args=(tweets_queue, data_queue))
        process.start()
        processes.append(process)
    
    tweets = get_tweets(session, chunk_size)
    count = 1
    while tweets:
        logging.warning("Processing tweets... (%s)", count*chunk_size)
        count += 1

        for chunk in chunk_tweets(tweets, processes_count):
            tweets_queue.put(chunk)

        returns = []
        for process in processes:
            returns.extend(data_queue.get())

        corpus.extend(returns)
        tweets = get_tweets(session, chunk_size)
    
    for process in processes:
        process.terminate()
        process.join()
    
    model = Word2Vec(min_count=1, epochs=1, vector_size=100)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=30, report_delay=1)
    model.save(str(WORD2VEC_MODEL_PATH))
    
    word_vectors = model.wv
    word_vectors.save(str(WORD2VEC_WV_PATH))
    del model
    
#    vectors = vectorize(word_vectors)
#    n_clusters = 100
#    km = kmeans_clusters(vectors, n_clusters, True)
#    clusters = compose_words_clusters(word_vectors, km, n_clusters)
#    save_clusters(clusters, session)


def test_kmeans():
    word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH))
    vectors = vectorize(word_vectors)
    k_clusters = range(2, K_CLUSTERS)
    
    min_sil = 1
    max_sil = -1
    mean_sil = 0

    data = {}

    for k in k_clusters:
        measurement = {'mean': 0, 'min': 1, 'max': -1}
        for step in range(STEPS):
            km = kmeans_clusters(vectors, k, False)
            silhouette = silhouette_score(vectors, km.labels_)
            measurement['min'] = float(min(measurement['min'], silhouette))
            measurement['max'] = float(max(measurement['max'], silhouette))
            measurement['mean'] += silhouette/STEPS
        data[k] = measurement

    with open(KMEANS_DATA_PATH, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def plot_kmeans_results():
    with open(KMEANS_DATA_PATH, 'r') as f:
        data = json.load(f)

    x = range(2, K_CLUSTERS)

    mean = []
    mins = []
    maxs = []

    value_measurement = {'value': -1, 'k': 0}
    max_values = {'mean': value_measurement, 'min': value_measurement, 'max': value_measurement}
    content = ['mean', 'min', 'max']


    for k in data:
        mean.append(data[k]['mean'])
        mins.append(data[k]['min'])
        maxs.append(data[k]['max'])
        
        for c in content:
            if data[k][c] > max_values[c]['value']:
                max_values[c]['value'] = data[k][c]
                max_values[c]['k'] = k

    print(max_values)

    plt.plot(x, mean, 'r', label = "Mean")
    plt.plot(x, mins, 'b', label = "Min")
    plt.plot(x, maxs, 'g', label = "Max")
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('silhouette')
    plt.title('k - silhouette')
    plt.savefig(KMEANS_GRAPH_PATH)
    plt.show()

# download stopwords
nltk.download('stopwords')

EPOCHS = 10
STEPS = 5
K_CLUSTERS = 21

if __name__ == '__main__':
    main()
    test_kmeans()
    plot_kmeans_results()

