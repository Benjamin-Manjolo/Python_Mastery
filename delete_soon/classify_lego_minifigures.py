import os
import math
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = 'C:/Users/Benjamin/Desktop/Projects/delete_soon/Dataset/star-wars/'
names = ["YODA","LUKE SKYWALKER","R2-D2","MACE WINDU","GENERAL GRIEVOUS"]

tf.random.set_seed(1)

train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'val')
test_dir = os.path.join(BASE_DIR, 'test')

# Generate batches of tensor image data with real-time data augmentation.
# The data will be looped over (in batches) indefinitely.
# preprocessing_function
# rescale=1./255
train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2, 
shear_range=0.2, 
zoom_range=0.2,  
horizontal_flip=True)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_batches = train_gen.flow_from_directory(
    train_dir,
    target_size=(256,256),
    class_mode="sparse",
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names
)

val_batches = valid_gen.flow_from_directory(
    val_dir,
    target_size=(256,256),
    class_mode="sparse",
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names
)

test_batches = test_gen.flow_from_directory(
    test_dir,
    target_size=(256,256),
    class_mode="sparse",
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

train_batch = train_batches[0]
print(train_batch[0].shape)
print(train_batch[1])
test_batch = test_batches[0]
print(test_batch[0].shape)
print(test_batch[1])

def show(batch,pred_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap='rgb')
        # The labels are numeric class indices.
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += " / Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()

show(test_batch)

model = keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3),strides=(1,1),padding="valid",activation="relu",input_shape=(256,256,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(5))
print(model.summary())

#loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics=["accuracy"]

model.compile(optimizer=optim,loss=loss,metrics=metrics)

#training
epochs = 10

#callbacks
history = model.fit(train_batches,epochs=epochs,validation_data=val_batches,verbose=2)

model.save("lego_minifigures.h5")

#plot loss and acc
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="train_loss")
plt.plot(history.history["val_loss"],label="val_loss")
plt.grid()
plt.legend()