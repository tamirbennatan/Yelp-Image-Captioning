import numpy as np
import pandas as pd 

import argparse


from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical

# read command-line args, to see if the user wants to just run a sample
argparser = argparse.ArgumentParser()
argparser.add_argument("--epochs", dest = "epochs", default= 20, type = int,
                 help="Indicate how many training epochs you want")
argparser.add_argument("--lr", dest = "learning_rate", default= .01, type = float,
                 help="Learning rate for training")
argparser.add_argument("--datadir", dest = "datadir", type = str,
                 help = "Relative path where data binary files are stored ")
argparser.add_argument("--modeldir", dest = "modeldir", type = str,
                 help = "Relative path where pickled models should be saved.")
argparser.add_argument("--historydir", dest = "historydir", type = str,
                 help = "Relative path where training hitory should be saved.")
argparser.add_argument("--patience", dest = "patience", type = int, default = 2,
                 help = "Early stopping patience.")

VOCAB_SIZE = 30212

def get_model(embedding_matrix):
        # input 1: photo features
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    # add a dense layer on top of that, with ReLU activation and random dropout
    drop1 = Dropout(0.5)(inputs_photo)
    dense1 = Dense(256, activation='relu')(drop1)

    #input 2: caption sequence
    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, 300,
                    mask_zero = True, trainable = False,
                    weights=[embedding_matrix])(inputs_caption)
    drop2 = Dropout(0.5)(embedding)
    lstm1 = LSTM(256)(drop2)

    #decoder model
    merged = add([dense1, lstm1])
    dense2 = Dense(256, activation='relu')(merged)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(dense2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs_photo, inputs_caption], outputs=outputs)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return(model)

"""
Load the training and validation data
"""
def load_npy(path):
    with open(path, "rb") as handle:
        arr = np.load(path)
    handle.close()
    return(arr)




































