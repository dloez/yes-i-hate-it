"""Process tweets and generate bag of worlds"""
import re
import sys
import logging
import multiprocessing as mp
import nltk
import stanza
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from unidecode import unidecode
from nltk.corpus import stopwords
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

from yes_i_hate_it.gather_tweets import Tweet

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import CLUSTER_LOG_FILE
from yes_i_hate_it.config import WORD2VEC_MODEL_PATH, WORD2VEC_WV_PATH


#class Word(BASE):
#    """Word base clase to interact with database"""
#    __tablename__ = 'word'
#
#    word = Column(String, primary_key=True)
#    cluster_id = Column(Integer, ForeignKey("cluster.id"), unique=False)
#    cluster = relationship('Cluster', back_populates='words')


#class Cluster(BASE):
#    """Cluster base clase to interact with databse"""
#    __tablename__ = 'cluster'
#
#    id = Column(Integer, primary_key=True)
#    words = relationship('Word', back_populates='cluster')


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


def create_clusters(words):
    """Create clusters"""
    word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH), mmap='r')
    return clusters


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
    x_cluster = []
    n_elements = []
    means = []
    for i, cluster in enumerate(clusters):
        x_cluster.append(i+1)
        #print(clusters[cluster]['words'])
        n_elements.append(len(clusters[cluster]['words']))
        means.append(clusters[cluster]['mean_similarity'])

    normalize_n = [float(i)/max(n_elements) for i in n_elements]

    plt.plot(x_cluster, normalize_n, 'r', label='n elements')
    plt.plot(x_cluster, means, 'b', label='means')
    plt.legend(loc='upper left')
    plt.xlabel('Cluster')
    plt.ylabel('n elements')
    plt.savefig('data/elem_sim.png')
    plt.clf()

    plt.plot(x_cluster, n_elements)
    plt.xlabel('Cluster')
    plt.ylabel('n elements')
    plt.savefig('data/elements.png')
    plt.clf()

    plt.plot(x_cluster, means)
    plt.xlabel('Cluster')
    plt.ylabel('mean similarity')
    plt.savefig('data/similarity.png')
    plt.clf()
    

def save_clusters(clusters, session):
    word_storage = []
    cluster_storage = []
    for i, cluster in enumerate(clusters):
        new_cluster = Cluster()
        for word in clusters[cluster]['words']:
            new_word = Word(word=word, cluster=new_cluster)
            word_storage.append(new_word)
        session.add_all(word_storage)
        word_storage = []
        session.add(new_cluster)
        session.commit()


def vectorize(corpus, model_vector_size, keyed_vector):
    """Generate vectors from copus (tweets) using word2vec model"""
    features = []
    for words in corpus:
        zero_vector = np.zeros(model_vector_size)
        vectors = []
        for word in words:
            if word in keyed_vector:
                try:
                    vectors.append(keyed_vector[word])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
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
    return km, km.labels_


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
    processes_count = mp.cpu_count()
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
    model.save(str(WORD2VEC_MODEL_PATH))

    word_vectors = model.wv
    word_vectors.save(str(WORD2VEC_WV_PATH))
    model_vector_size = model.vector_size
    del model

    vectors = vectorize(corpus, model_vector_size, word_vectors)
    clusters = kmeans_clusters(vectors, 100, True)
    print(clusters)

    # delete repeated words
    #clean_words = set()
    #for sentence in processed_tweets:
    #    for word in sentence:
    #        clean_words.add(word)
    #del processed_tweets

    #clean_words = list(clean_words)
    #clusters = create_clusters(clean_words)
    #save_clusters(clusters, session)


# download stopwords
nltk.download('stopwords')

EPOCHS = 10

if __name__ == '__main__':
    main()
