#The Goal of this project is to classfy whether a movie review is postive or negative.


import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# access most frequent 10000 usage
(train_data, train_labels), (test_data,test_labels) = data.load_data(num_words=10000)
# A series of numbers that maps to words, so we need to code a function to make the words readable. 
# Gives a dictionary with number frequency and its number in it.
word_index = data.get_word_index()

#Deal with speacial characters fulfill the spaces.
#This would not change the mapping on words 
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0  #To fill the empty spaces to make each review to length 200.
word_index['<START>'] = 1 
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3


# Now, we want to print out word rather than index
#Since word_index has Key = words, value = frequency, we reverse the order.

reversed_word_index = {v:k for k,v in word_index.items()}

# preprocess to ensure that each data length is 200.
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
    maxlen = 200, 
    padding="post",
    truncating = "post",
    value = word_index['<PAD>'])

test_data = keras.preprocessing.sequence.pad_sequences(test_data, 
    maxlen = 200, 
    padding="post",
    truncating = "post",
    value = word_index['<PAD>'])

# the decode function
def decode(text):
    return " ".join(reversed_word_index.get(i,"<?>") for i in text)

print(decode(test_data[0]))


#Make the model:
model = keras.Sequential()








