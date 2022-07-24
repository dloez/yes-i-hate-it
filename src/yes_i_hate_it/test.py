import tensorflow as tf
from tensorflow.keras import Sequential, layers
import os

model = Sequential([
    layers.Embedding(10, 5, input_length=200),
    layers.Dropout(0.2),
    layers.LSTM(units=64, return_sequences=True),
    layers.LSTM(units=64),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print(model.summary())
