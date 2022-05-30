"""Neural network train"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from yes_i_hate_it.generate_bow import Word

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import FIGURE_FILE_PATH, MODEL_FILE_PATH, EPOCHS
from yes_i_hate_it.process_dataset import load_dataset, from_text_to_array


def create_model(total_words):
    """Create network model"""
    inputs = tf.keras.layers.Input(shape=(total_words,))
    hidden1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)(inputs)
    hiddenf = tf.keras.layers.Dense(100, activation=tf.nn.relu)(hidden1)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(hiddenf)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Accuracy()
        ]
    )

    return model


def plot_history(history):
    """Plot train history in figure"""
    np_range = np.arange(0, EPOCHS)
    plt.figure()
    plt.plot(np_range, history.history["loss"], label="train_loss")
    plt.plot(np_range, history.history["val_loss"], label="val_loss")
    plt.title("Trainning Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(FIGURE_FILE_PATH)


def load_model():
    """Load trained model"""
    return tf.keras.models.load_model(MODEL_FILE_PATH)


def main():
    """Main function to execute nerual network"""
    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    (train_data, train_labels), (test_data, test_label) = load_dataset()
    results = ['football', 'not football']

    model = create_model(session.query(Word).count())
    model.summary()
    history = model.fit(
        train_data, train_labels,
        validation_split=0.15,
        epochs=10000,
        shuffle=True
    )

    # model = load_model()

    test = model.evaluate(test_data, test_label)
    print(test)
    plot_history(history)
    model.save(MODEL_FILE_PATH)

    text = from_text_to_array(session, "el atletico es lo mejor")
    prediction = model.predict(np.array([text]))
    print(f'{results[0]}: {prediction[0][0]}')
    print(f'{results[1]}: {prediction[0][1]}')


if __name__ == '__main__':
    main()
