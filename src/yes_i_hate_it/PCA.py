import sys
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sqlalchemy import Column, String, Integer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from yes_i_hate_it.process_dataset import load_dataset

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import BASE


class PCAWords(BASE):
    """Tweet base clase to interact with database"""
    __tablename__ = 'pca_words'

    id = Column(Integer, primary_key=True)
    text = Column(String, unique=False)


# pylint:disable = invalid-name
def process_PCA(session):
    """Process dataset with PCA algorithm"""
    if 'yes_i_hate_it.generate_bow.process_text' not in sys.modules:
        from yes_i_hate_it.generate_bow import Word
    (train_x, _), (test_x, _) = load_dataset()

    print("loaded train ", train_x.shape)
    train = np.concatenate((train_x, test_x), axis=0)
    del train_x, test_x
    print("Train set ", train.shape)
    pca = PCA(.95)

    # train = train_x + test_x
    print("loaded dataset")
    print("fitting PCA")
    pca.fit(train)
    print(pca.n_components_)
    print("transform PCA")
    train = pca.transform(train)
    print('shape ', train.shape)
    print('train ', train)
    print("type ", type(train))

    most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)]

    initial_feature_names = session.query(Word).all()
    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.n_components_)]
    # words = [word.text for word in most_important]
    print("length ", len(most_important), " ", len(most_important_names))
    del most_important, initial_feature_names
    for word in set(most_important_names):
        new_word = PCAWords(text=word.text)
        session.add(new_word)
    session.commit()

    with open('data/pca.pickle', 'wb') as file:
        pickle.dump(pca, file)


def read_PCA(session):
    """Read PCA"""
    if 'yes_i_hate_it.generate_bow.process_text' not in sys.modules:
        from yes_i_hate_it.generate_bow import Word
    with open("data/pca.pickle", 'rb') as f:
        pca = pickle.load(f)

    print(pca.n_components_)
    print(type(pca))

    most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)]
    print(len(pca.components_[0]))
    initial_feature_names = session.query(Word).all()
    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.n_components_)]

    print(most_important)

    # print(len(most_important), len(most_important_names))

    # for word in most_important_names:
    #     print(word.text)

    # for word in most_important_names:
    #     new_word = PCAWords(text=word.text)
    #     session.add(new_word)
    # session.commit()
        # try:
        #     session.commit()
        # except IntegrityError:
        #     # logging.info("%s repeated, skipping...", word)
        #     session.rollback()


def test_PCA():
    # train = np.array(((1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)))
    train = np.array(
        (
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 1, 1, 1, 1, 1),
            (0, 1, 0, 0, 1, 0, 1, 0, 1, 1),
            (0, 1, 1, 0, 1, 0, 1, 0, 0, 1),
            (1, 0, 1, 0, 1, 0, 1, 0, 1, 1),
            (1, 1, 0, 1, 1, 1, 0, 1, 1, 1),
            (1, 1, 1, 1, 0, 0, 1, 1, 1, 0),
            (0, 1, 1, 0, 0, 1, 1, 1, 0, 0),
            (1, 1, 0, 1, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 1, 1, 0, 1, 1, 1, 0),
            (0, 0, 1, 0, 0, 0, 0, 1, 1, 1),
            (1, 1, 1, 0, 1, 0, 1, 0, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 0, 0, 1),
            (0, 0, 1, 0, 0, 0, 0, 1, 0, 0),
            (1, 1, 0, 0, 1, 1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            (1, 0, 1, 1, 0, 0, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 1, 0, 0, 1, 1),
            (1, 0, 1, 1, 1, 0, 1, 0, 1, 1),
            (1, 1, 0, 1, 1, 0, 0, 1, 0, 1)
        )
    )

    pca = PCA(n_components=2)
    print("fitting")
    pca.fit(train)
    print(train.shape)
    train = pca.transform(train)
    print(train.shape)
    print(pca.components_)
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)]

    initial_feature_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.n_components_)]
    print(most_important)
    print(most_important_names)


def main():
    """main function"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    BASE.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    process_PCA(session_maker())
    # read_PCA(session_maker())
    

if __name__ == '__main__':
    main()
    # test_PCA()
