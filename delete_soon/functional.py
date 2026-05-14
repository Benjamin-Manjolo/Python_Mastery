import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np

# a        a       a     b       a
# |        |        \   /       /  \
# b        b          c        b    c
# |       / \         |        \   /
# C      c   d        d          d

# model: Sequential: one input, one output
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

print(model.summary())

#functional API 
inputs = keras.Input(shape=(28,28))
flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128,activation='relu')
dense2 = keras.layers.Dense(10)
dense2_2 = keras.layers.Dense(1)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)
outputs2 = dense2_2(x)

model = keras.Model(inputs = inputs, outputs = [outputs,outputs2],name='functional_model')
print(model.summary())


new_model = keras.models.Sequential()
for layer in model.layers:
    new_model.add(layer)