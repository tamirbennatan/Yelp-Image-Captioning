from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import pickle
import pdb
import io
import base64

import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from sequence_candidate import SequenceCandidate

"""
Fixed variables:
    - PHOTO_MODEL: model used to convert an image into a (4096,) dimensional vector
    - CAPTION_MODEL: model that takes in a sequence and a photo, and predicts the next word
    - TOKENIZER: word tokenizer used for converting sequences to strings and back
"""

PHOTO_MODEL = "models/photo_processor/model.h5"
# inject model
CAPTION_MODEL = "models/caption_generator/model_inject-date_5-16-15-45-ep030-loss5.009_lr-0.010000_patience-3.h5"
# merge model
# CAPTION_MODEL = "models/caption_generator/model_merge-date_5-18-14-40-ep016-loss4.704_lr-0.010000_patience-3.h5"
TOKENIZER = "models/tokenizer/tokenizer.pkl"

# define application.
app = Flask(__name__)

"""
Functions to load the models into memory. 
This is done only once at the start-up of the server, as it is I/O expensive. 
"""
def load_image_processor():
    global image_processor
    image_processor = load_model(PHOTO_MODEL)
    global graph
    graph = tf.get_default_graph()
def load_caption_model():
    global caption_model
    caption_model = load_model(CAPTION_MODEL)
    global graph2
    graph2 = tf.get_default_graph()

def load_tokenizer():
    global tokenizer
    with open(TOKENIZER, "rb") as handle:
        tokenizer = pickle.load(handle)
    handle.close()
    # dictionary of {index:word} pairs. 
    global reverse_tokenizer
    reverse_tokenizer = {index: word for word,index in tokenizer.word_index.items()}

def load_validation():
    global valid_set
    valid_set = pd.read_csv("static/valid_set/valid_ids.csv")


"""
Prepare an image to be processed by VGG16
"""
def prepare_image(image_path, target_size = (224,224)):
    # load the image
    image = load_img(image_path, target_size =target_size )
    # convert to numpy array
    image = img_to_array(image)
    # add a fourth dimension - to be predicted upon
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess for VGG16
    image = preprocess_input(image)
    return(image)

"""
Extract photo features from photo
"""
def extract_photo_features(image_path):
    # preprocess image data
    image = prepare_image(image_path)
    # use photo embedding model to predict features
    global graph
    with graph.as_default():
        features = image_processor.predict(image)
        return(features)

def generate_predictions_beam(image_path, width, num_neighbors,
                                 top_n = 3, end_idx = 2, max_length = 15, ignore_idx = [4,61,345], alpha = .6):
    global graph2
    # isolate the photo features
    photo_features = extract_photo_features(image_path)
    # keep track of the accepted sequences
    accepted_sequences = []
    # keep track of the current population
    population = []
    # add a start sequence to the population
    start_sequence = SequenceCandidate.template_seq(ignore_idx = ignore_idx, alpha = alpha)
    population.append(start_sequence)
    for i in range(max_length - 1):
        tmp = []
        for cand_seq in population:
            # pdb.set_trace()
            with graph2.as_default():
            # get the prediction of the next word 
                pred = caption_model.predict([photo_features, cand_seq._seq.reshape(1,-1)], verbose=0)[0]
            # sort the predicted next words by their probabilities
            pred_argsort = pred.argsort()
            # add candidates for each of the <num_neighbors> neighbors
            for next_idx in pred_argsort[-num_neighbors:]:
                # if we're starting to repeat bigrams, accept the current candidate
                if (cand_seq.final_token(), next_idx) in cand_seq._bigrams:
                    accepted_sequences.append(cand_seq)
                    continue
                # add the predicted word to get a new candidate
                next_prob = pred[next_idx]
                new_candidate = cand_seq.add_token(next_idx,next_prob)
                # if the next suggested token is <endseq>, add to accepted_sequences
                if next_idx == end_idx:
                    accepted_sequences.append(new_candidate)
                else:
                    tmp.append(new_candidate)
        # prune the population to keep only the top <width> candidates. 
        try:
            population = sorted(tmp)[-width:]
        except:
            # fewer than <width> individuals remain - stop growing tree and keep curren partial sequences
            population = tmp
            break
    # add current population to accepted sequences
    accepted_sequences = sorted(accepted_sequences + population, reverse = True)
    # build output JSON data 
    num_accepted = 0
    values = []
    x = []
    y = []
    for acc_seq in accepted_sequences:
        # convert current sequence to words
        seq_string = acc_seq.to_words(reverse_tokenizer,end_idx)
        # if its not already in one of the word lists, accept it.
        if seq_string not in y:
            y.append(seq_string)
            x.append(acc_seq.probsum())
            num_accepted += 1
            # if you've already accepted <top_n>, you're done
            if num_accepted >= top_n:
                break

    output = [{"x":x[::-1], 
                "y":"", 
                "type":'bar', 
                'orientation':'h', 
                'text': y[::-1], 
                'textposition': 'auto',
                'marker': {
                    'color': 'rgb(158,202,225)',
                    'opacity': 0.6,
                    'line': {
                      'color': 'rbg(8,48,107)',
                      'width': 1.5
                    }
                }
            }]
    return output

@app.route('/', methods=['GET'])
def index():
    # sample 9 images and captions
    smpl = valid_set.sample(9)
    # isolate captions and paths, then reshape to 2d array
    paths = smpl.photo_id.values
    captions = smpl.caption.values
    # put in a dictionary
    photo_captions = {
        "first": list(zip(paths[:3], captions[:3])),
        "second": list(zip(paths[3:6], captions[3:6])),
        "third": list(zip(paths[6:9], captions[6:9]))
    }
    # Main page
    return render_template('index.html',photo_captions=photo_captions)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename('pic.' + f.filename.rsplit('.', 1)[1].lower()))
        f.save(file_path)

        # Make prediction
        preds = generate_predictions_beam(file_path, width = 5, num_neighbors = 3)

        return jsonify(preds)

    return None

@app.route('/sample_image', methods=['GET', 'POST'])
def sample_image():
    if request.method == 'POST':
        # get the file path the user asked for
        sample_path = request.data.decode('utf-8')[1:]
        # Make prediction
        preds = generate_predictions_beam(sample_path, width = 5, num_neighbors = 3)
        return jsonify(preds)
    return None

if __name__ == "__main__":
    # load all the models to memory
    load_image_processor()
    load_caption_model()
    load_tokenizer()
    load_validation()
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()



