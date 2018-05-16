import numpy as np


from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, RepeatVector, Masking
from keras.layers.merge import add, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical

# global variables
VOCAB_SIZE = 30212
EMBED_SIZE = 300


def get_merge_model(embedding_matrix,trainable = True):
        # input 1: photo features
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    # A first dense layer
    dense = Dense(4096, activation = 'relu')(inputs_photo)
    # add a dense layer on top of that, with ReLU activation and random dropout
    drop1 = Dropout(0.5)(dense)
    dense1 = Dense(256, activation='relu')(drop1)

    #input 2: caption sequence
    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE,
                    mask_zero = True, trainable = trainable,
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

def get_inject_model(embedding_matrix, trainable = True):
    # input 1: photo features
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    # add a dense layer on top of that, with ReLU activation and random dropout
    drop1 = Dropout(0.5)(inputs_photo)
    dense1 = Dense(EMBED_SIZE, activation='relu')(drop1)
    # add time dimension so that this layer output shape is (None, 1, embed_size)
    cnn_feats = Masking()(RepeatVector(1)(dense1))

    #input 2: caption sequence
    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE,
                    mask_zero = True, trainable = False,
                    weights=[embedding_matrix])(inputs_caption)
    drop2 = Dropout(0.5)(embedding)
    # merge the models: decoder model
    # Ouput shape should be (None, maxlen + 1, embed_size)
    merged = concatenate([cnn_feats, drop2], axis=1)
    # now feed the concatenation into a LSTM layer (many-to-many)
    lstm_layer = LSTM(units=EMBED_SIZE,
                      input_shape=(15 + 1, EMBED_SIZE),   # one additional time step for the image features
                      return_sequences=False,
                      dropout=.5)(merged)

        # create a fully connected layer to make the predictions
    outputs = Dense(units=VOCAB_SIZE,activation='softmax')(lstm_layer)

    model = Model(inputs=[inputs_photo, inputs_caption], outputs=outputs)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return(model)