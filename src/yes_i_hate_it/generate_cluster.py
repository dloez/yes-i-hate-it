"""Process tweets and generate bag of worlds"""
import re
import sys
# import random
import logging
# import multiprocessing as mp
import nltk
import stanza
from gensim.models import Word2Vec, KeyedVectors
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


class Word(BASE):
    """Word base clase to interact with database"""
    __tablename__ = 'word'

    word = Column(String, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("cluster.id"), unique=False)
    cluster = relationship('Cluster', back_populates='words')


class Cluster(BASE):
    """Cluster base clase to interact with databse"""
    __tablename__ = 'cluster'

    id = Column(Integer, primary_key=True)
    words = relationship('Word', back_populates='cluster')


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


def create_clusters(words):
    """Create clusters"""
    word_vectors = KeyedVectors.load(str(WORD2VEC_WV_PATH), mmap='r')
    # c1 = {'words': [], 'mean_similarity': 0}
    clusters = {} #{'c1': c1}
    clustered_words = {} #{'porterazos': 'c1'}
    new_cluster = 1
    for word in words:
        word_similarities = max_normalized_sim(word_vectors, word)
        best_similarity = 0
        for sim in word_similarities:
            similarity = sim[1]
            if similarity > best_similarity:
                best_similarity = similarity
                word_to_append = sim[0]

        # if word in clustered_words:
        #     word_cluster = clustered_words[word]
        #     if word_to_append in clustered_words:
        #         if word_cluster == clustered_words[word_to_append]:
        #             continue
        # pylint: disable = line-too-long
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
    

def generate_boc(clusters, session):
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

    processed_tweets = []
    tweets = get_tweets(session, 10)

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
    clusters = create_clusters(clean_words)
    generate_boc(clusters, session)


# download stopwords
nltk.download('stopwords')
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

EPOCHS = 10

if __name__ == '__main__':
    main()
