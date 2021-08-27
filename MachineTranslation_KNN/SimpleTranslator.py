#!/usr/bin/env python

import pdb
import pickle
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy
import sklearn
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import TweetTokenizer

from utils import (cosine_similarity, get_dict,
                   process_tweet)


def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """

    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()
    Y_l = list()

    # get the words (the keys in the dictionary) and store in a set()
    english_set = set(english_vecs.keys())
    french_set = set(french_vecs.keys())

    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():
        if fr_word in french_set and en_word in english_set:
            en_vec = english_vecs[en_word]
            fr_vec = french_vecs[fr_word]
            X_l.append(en_vec)
            Y_l.append(fr_vec)

    X = np.vstack(X_l)
    Y = np.vstack(Y_l)

    return X, Y

def compute_loss(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    m = X.shape[0]
    diff = np.dot(X,R) - Y
    diff_squared = np.square(diff)
    sum_diff_squared = np.sum(diff_squared)
    loss = sum_diff_squared/m
    return loss

def compute_gradient(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a scalar value - gradient of the loss function L for given X, Y and R.
    '''
    m = X.shape[0]
    gradient = 2/m*np.dot(X.T,(np.dot(X,R) - Y))
    return gradient

def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    R = np.random.rand(X.shape[1], X.shape[1])
    for i in range(train_steps):
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        gradient = compute_gradient(X, Y, R)
        R -= learning_rate*gradient
    return R

def nearest_neighbor(v, candidates, k=1):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []
    for row in candidates:
        cos_similarity = cosine_similarity(v,row)
        similarity_l.append(cos_similarity)
    sorted_ids = np.argsort(similarity_l)
    k_idx = sorted_ids[-k:]
    return k_idx

def test_vocabulary(X, Y, R):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''
    pred = np.dot(X,R)
    num_correct = 0
    for i in range(len(pred)):
        pred_idx = nearest_neighbor(pred[i], Y, k=1)
        if pred_idx == i:
            num_correct += 1
    accuracy = num_correct/X.shape[0]
    return accuracy

en_embeddings_subset = pickle.load(open("en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("fr_embeddings.p", "rb"))

en_fr_train = get_dict('en-fr.train.txt')
en_fr_test = get_dict('en-fr.test.txt')


X_train, Y_train = get_matrices(
    en_fr_train, fr_embeddings_subset, en_embeddings_subset)
X_val, Y_val = get_matrices(
    en_fr_test, fr_embeddings_subset, en_embeddings_subset)

R_train = align_embeddings(X_train, Y_train, train_steps=500, learning_rate=0.8)

acc = test_vocabulary(X_val, Y_val, R_train)
print(f"accuracy on test set is {acc:.3f}")

acc = test_vocabulary(X_train, Y_train, R_train)
print(f"accuracy on train set is {acc:.3f}")
