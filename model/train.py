import numpy as np
import pandas as pd

import argparse
import datetime
import pickle
import gc

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
argparser.add_argument("--lr", dest = "lr", default= .01, type = float,
                 help="Learning rate for training")
argparser.add_argument("--datadir", dest = "datadir", type = str,
                 help = "Relative path where data binary files are stored ")
argparser.add_argument("--embeddingdir", dest = "embeddingdir", type = str,
                 help = "Relative path where the embedding matrix binary file is stored.")
argparser.add_argument("--modeldir", dest = "modeldir", type = str,
                 help = "Relative path where pickled models should be saved.")
argparser.add_argument("--historydir", dest = "historydir", type = str,
                 help = "Relative path where training hitory should be saved.")
argparser.add_argument("--patience", dest = "patience", type = int, default = 2,
                 help = "Early stopping patience.")
# argparser.add_argument("--batch", dest = "batch", type = int, default = 128,
#                  help = "Batch size.")

args = argparser.parse_args()
# store how many training epochs
epochs = args.epochs
# store learning rate
lr = args.lr
# store the data directory
datadir = args.datadir
# store the embedding matrix location
embeddingdir = args.embeddingdir
# store the model directory
modeldir = args.modeldir
# store the history directory
historydir = args.historydir
# store the training patience
patience = args.patience
# store the batch size
# batch = args.batch

VOCAB_SIZE = 30212


def get_model(embedding_matrix):
        # input 1: photo features
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    # A first dense layer
    dense = Dense(4096, activation = 'relu')(inputs_photo)
    # add a dense layer on top of that, with ReLU activation and random dropout
    drop1 = Dropout(0.5)(dense)
    dense1 = Dense(256, activation='relu')(drop1)

    #input 2: caption sequence
    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, 300,
                    mask_zero = True, trainable = True,
                    weights=[embedding_matrix])(inputs_caption)
    drop2 = Dropout(0.5)(embedding)
    lstm1 = LSTM(256)(drop2)
    #merge the LSTM and CNN outputs, and slap a few dense layers on top. 
    merged = add([dense1, lstm1])
    dense2 = Dense(256, activation='relu')(merged)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(dense2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs_photo, inputs_caption], outputs=outputs)
    sgd = SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return(model)

"""
Load the training and validation data
"""
def load_npy(path):
    with open(path, "rb") as handle:
        arr = np.load(path)
    handle.close()
    return(arr)


"""
Define checkpoints for model checkpoints, as well as early stopping
"""
today = datetime.datetime.now()
model_path = modeldir + 'model-date_%d-%d-%d-%d-ep{epoch:03d}-loss{loss:.3f}_lr-%f_patience-%d.h5' % (
    today.month, today.day, today.hour, today.minute, lr, patience)
# model checkpoint
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=False)
# early stopping
early_stopping = EarlyStopping(patience=patience)
"""
fit the model. keep track of the training history.
"""
print("Loading model and embedding matrix...")
embedding_matrix = load_npy(embeddingdir + "embedding_matrix.npy")
# initialize a model
model = get_model(embedding_matrix)
del embedding_matrix
gc.collect()
print("done.")
print()
print("Loading training data...")
# training data/labels
y_train = load_npy(datadir + "y_train.npy")
y_train = y_train.reshape((-1,))
X_train_photos = load_npy(datadir + "X_train_photos.npy")
X_train_captions = load_npy(datadir + "X_train_captions.npy")
print("done.")
print()
print("Loading validation data...")
y_valid = load_npy(datadir + "y_valid.npy")
y_valid = y_valid.reshape((-1,))
X_valid_photos = load_npy(datadir + "X_valid_photos.npy")
X_valid_captions = load_npy(datadir + "X_valid_captions.npy")
# save the number of examples for later
NUM_EXAMPLES = X_train_photos.shape[0]

history = model.fit([X_train_photos, X_train_captions], y_train, epochs=epochs, verbose=1,
    callbacks=[checkpoint,early_stopping], 
    validation_data=([X_valid_photos, X_valid_captions], y_valid))

"""
Save the training history
"""
history_path = historydir + "history-date_%d-%d-%d-%d.pkl" % (today.month, today.day, today.hour, today.minute)
with open(history_path, "wb") as handle:
    pickle.dump(history.history, handle)
handle.close()
