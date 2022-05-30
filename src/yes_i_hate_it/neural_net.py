"""Neural network train"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def main():
    # (train_data, train_labels), (test_data, test_label) = load_data()
    results = ['football', 'not football']
    inputs = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(28, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # model = tf.keras.Sequential()
    # model.add(keras.Input(shape=(4,)))
    # model.add(keras.layers.Dense(32, activation="relu"))
    # model.add(keras.layers.Dense(2))
    
    print(model.weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
    print("#############################################")
    print(model.weights)
    # model.build()
    model.summary()
    # print(tf.keras.__version__)


if __name__ == '__main__':
    main()
