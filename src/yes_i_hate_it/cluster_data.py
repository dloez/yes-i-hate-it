"""Vectorize dataset to make clusters"""
import re
import stanza
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from yes_i_hate_it.config import TWEETS_DB_PATH

from yes_i_hate_it.gather_tweets import Tweet

def create_sentences():
    """Process text to create sentence for Word2Vec"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    tweets = session.query(Tweet).filter(Tweet.processed==True).all()
    text = []
    for tweet in tweets:
        txt = tweet.text
        txt = re.sub(r'(https|ftp|http):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', '', txt)
        txt = unidecode(txt)
        txt = txt.lower() # lowercase text
        txt = re.sub(r'\d+', '', txt) # remove numbers
        txt = re.sub(r'[^\w\s]', ' ', txt) # remove non char/spaces
        txt = txt.replace('_', '') # remove _
        txt = txt.split()
        text.append(txt)
    print(text)

    # nltk.download('stopwords')
    # nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

    print(text)
    # words that have no meanings
    stop_words = [*stopwords.words('spanish'), *stopwords.words('english')]

    processed_sentences = []

    for sentence in text:
        txt = []
        for word in sentence:
            if word not in stop_words:
                txt.append(word)
        processed_sentences.append(txt)

    return processed_sentences
    # doc = nlp(text)

sentences = create_sentences()

# sentences = [['me', 'gusta', 'el', 'futbol'], ['vaya', 'dia', 'de', 'mierda', 'que', 'llevo']]

model =  Word2Vec(min_count=1, epochs=1, vector_size=100)
# gensim.models.Word2Vec(sentences, iter=100, size=200, workers = 4)
model.build_vocab(sentences)
# model.train(data)

# v1 = model.wv['futbol']

sim = model.wv.most_similar('futbol')

print(sim)


# print(f"futbol-pie: {model1.wv.similarity('futbol', 'pie')}")
# print(f"balon-pie: {model.accuracy('balon pie')}")
