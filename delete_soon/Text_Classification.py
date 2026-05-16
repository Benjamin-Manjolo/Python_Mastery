import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

#https://www.kaggle.com/c/cn/nlp-getting-started : NLP Disaster Tweets
df = pd.read_csv("data/twitter_train.csv")

df.shape
df.head()

print((df.target == 1).sum()) #Disaster
print((df.target == 0).sum()) #No disaster

#preprocessing
import re
import string

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"",text)

#https://stackoverflow.com/questions/3429875/how-to-remove-punctuation-from-a-string-in-python-3-x-using-translate-table
def remove_punc(text):
    translator = str.maketrans("","",string.punctuation)
    return text.translate(translator)

string.punctuation

pattern =re.compile(r"https?://\S+|www\.\S+")
for t in df.text:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"",t))
    if len(matches) > 0:
        break

df["text"] = df.text.map[remove_URL] #map(lambda x: remove_URL(x))
df["text"] = df.text.map[remove_punc] #map(lambda x: remove_punc(x))

#remove stop words
#pip install nltk
import nltk
nltk.download("stopwords")  
from nltk.corpus import stopwords

#stop words: a stop word is a commonly used word (such as "the", "a", "an", "in") that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.
#has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.

stop = set(stopwords.words("english"))

#https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-in-python
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)
stop

df["text"] = df.text.map[remove_stopwords] #map(lambda x: remove_stopwords(x))
df.text

from colllections import Counter

#count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(df.text)
len(counter)

counter
counter.most_common(5)
num_unique_words = len(counter)

#spliit dataset into training and validation set
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

#split text and labels
train_sentences = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_sentences = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()

train_sentences.shape, val_sentences.shape

#Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

#vectorize a text corpus by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences) #fit only training 

#each word has an index
word_index = tokenizer.word_index
word_index

train_sentences = tokenizer.texts_to_sequences(train_sentences)
val_sentences = tokenizer.texts_to_sequences(val_sentences)

print(train_sentences[10:15])
print(val_sentences[10:15])

#pad sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Max number of words in a sequence
max_length = 20

train_padded = pad_sequences(train_sentences, maxlen=max_length, padding="post",truncating="post")
val_padded = pad_sequences(val_sentences, maxlen=max_length, padding="post",truncating="post")
train_padded.shape,val_padded.shape

train_padded[10]

print(train_labels[10:15])
print(train_sentences[10])
print(train_padded[10])

#check reversing the indices
#flip (key,values)
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
reverse_word_index

def decode_sentence(text):
    return " ".join([reverse_word_index.get(idx,"?") for idx in sequence])
reverse_word_index

def decode(sequence):
    return " ".join([reverse_word_index.get(idx,"?") for idx in sequence])
decoded_text = decode(train_sequences[10])
print[train_sequences[10]]
print(decoded_text)

#Create LSTM model
from tensorflow.keras import layers
 
#Embeddind : https://www.tensorflow.org/ttorials/text/word_embeddings
#turns positive integers (indexes) into dense vectors of fixed size..(other approach couuld be one-hot-encoding)
#word embeddings give us a away to use efficient, dense representain in which similar words have
#a similar encoding, importantly,you do not have to specify this encoding by hand .An embedding is a
#dense vector   of floating point values 

model = keras.models.Seuential()
model.add(layers.Embedding(num_unique_words,32,input_length=max_length))

#The layer will take as input an integer matrix of size (batch,input-length),
#and the largest integer (i.e word index) in the input should be no larger than num_words (vocabulary words)
 
model.add(layers.LSTM(64,dropout=0.1))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]
model.compile[loss=loss,optimizer=optim,metrics=metrics]

model.fit(train_padded,train_labels,epochs=20,validation_data=(val_padded,val_labels),verbose=2)

predictions = model.predict(train_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print(train_sentences[10:20])
print(predictions[10:20])