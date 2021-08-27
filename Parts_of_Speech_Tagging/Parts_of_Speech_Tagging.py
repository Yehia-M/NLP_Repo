#!/usr/bin/env python

from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np

def create_dictionaries(training_corpus, vocab):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """


    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    prev_tag = '--s--'

    # use 'i' to track the line number in the corpus
    i = 0

    for word_tag in training_corpus:
        i += 1

        # Every 50,000 words, print the word count
        if i % 50000 == 0:
            print(f"word count = {i}")

        word, tag =  get_word_tag(word_tag,vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag

    return emission_counts, transition_counts, tag_counts

def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    Input:
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Output:
        accuracy: Number of times you classified a word correctly
    '''
    num_correct = 0
    all_words = set(emission_counts.keys())
    total = len(y)

    for word, y_tup in zip(prep, y):
        y_tup_l = y_tup.split()

        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:

            # Set the true POS label for this word
            true_label = y_tup_l[1]

        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue

        count_final = 0
        pos_final = ''

        if word in vocab:

            for pos in states:
                key = (pos,word)

                if key in emission_counts.keys():
                    count = emission_counts[key]
                    if count > count_final:
                        count_final = count
                        pos_final = pos

            if pos_final == true_label:
                num_correct += 1

    accuracy = num_correct / total
    return accuracy


def predict_only(word,emission_counts,vocab, states):
    '''
    Input:
        word: Will try to predict its PoS
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Output:
        pos_final: Predicted Part-of-Speech of the given word
    '''
    count_final = 0
    pos_final = ''
    if word in vocab:
        for pos in states:
            key = (pos,word)
            if key in emission_counts.keys():
                count = emission_counts[key]
                if count > count_final:
                    count_final = count
                    pos_final = pos
    else:
        print(word +" is not in vocab")

    return pos_final

def create_transition_matrix(alpha, tag_counts, transition_counts):
    '''
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: transition count for the previous word and tag
    Output:
        A: matrix of dimension (num_tags,num_tags)
    '''
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags,num_tags))
    trans_keys = set(transition_counts.keys())


    for i in range(num_tags):
        for j in range(num_tags):
            count = 0

            # Define the tuple (prev POS, current POS)
            key = (all_tags[i],all_tags[j])
            if key in transition_counts:
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]

            # Apply smoothing using count of the tuple, alpha,
            # count of previous tag, alpha, and total number of tags
            A[i,j] = (count + alpha)/(count_prev_tag + alpha*num_tags)

    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    '''
    Input:
        alpha: tuning parameter used in smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index.
               within the function it'll be treated as a list
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    '''
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    B = np.zeros((num_tags, num_words))
    emis_keys = set(list(emission_counts.keys()))

    for i in range(num_tags):
        for j in range(num_words):
            count = 0

            # Define the (POS tag, word) tuple for this row and column
            key =  (all_tags[i],vocab[j])
            if key in emission_counts:
                count = emission_counts[key]
            count_tag = tag_counts[all_tags[i]]

            # Apply smoothing and store the smoothed value
            # into the emission matrix B for this row and column
            B[i,j] = (count + alpha)/ (count_tag + alpha*num_words)

    return B


def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Input:
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''
    num_tags = len(tag_counts)
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    s_idx = states.index("--s--")

    for i in range(num_tags):

        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx,i] == 0:
            best_probs[i,0] = float('-inf')

        # For all other cases when transition from start token to POS tag i is non-zero:
        else:
            best_probs[i,0] = math.log(A[s_idx,i]) + math.log(B[i,vocab[corpus[0]]])

    return best_probs, best_paths


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    '''
    Input:
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''

    num_tags = best_probs.shape[0]

    for i in range(1, len(test_corpus)):

        # Print number of words processed, every 5000 words
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))

        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None
            for k in range(num_tags):
                # Calculate the probability
                prob = best_probs[k,i-1] + math.log(A[k,j]) + math.log( B[ j, vocab[ test_corpus[i] ] ] )

                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            best_probs[j,i] = best_prob_i
            best_paths[j,i] = best_path_i

    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, corpus, states):
    '''
    This function returns the best path.

    '''
    m = best_paths.shape[1]
    z = [None] * m
    num_tags = best_probs.shape[0]
    best_prob_for_last_word = float('-inf')
    pred = [None] * m

    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID)
    # with highest probability for the last word
    for k in range(num_tags):
        if best_probs[k,-1] > best_prob_for_last_word:
            best_prob_for_last_word = best_probs[k,-1]

            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k

    # Convert the last word's predicted POS tag from its unique integer ID
    # into the string representation using the 'states' list
    pred[m - 1] = states[k]


    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(m-1, 0, -1): # complete this line

        pos_tag_for_word_i = best_paths[z[i], i]
        z[i - 1] = pos_tag_for_word_i
        pred[i - 1] = states[pos_tag_for_word_i]
    return pred

def compute_accuracy(pred, y):
    '''
    Input:
        pred: a list of the predicted parts-of-speech
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:
        Accuracy of Viterbi Algorithm
    '''
    num_correct = 0
    total = 0

    for prediction, y in zip(pred, y):
        word_tag_tuple = y.strip().split('\t')
        if len(word_tag_tuple) != 2 :
            continue

        word, tag = word_tag_tuple[0],word_tag_tuple[1]
        if prediction == tag:
            num_correct += 1
        total += 1
    return num_correct/total

def find_word_pos(word,prep, pred):
    '''
    Input:
        word: Goal is to predict its PoS
        prep: all words in vocab
        pred: a list of the predicted parts-of-speech
    Output:
        predicion of word
    '''
    prediction = ''
    if word in prep:
        indx = prep.index(word)
        prediction = pred[indx]
    else:
        print("{} is not in vocab".format(word))

    #print('The prediction for {} is: {}'.format(word,prediction))
    return prediction


with open("WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

with open("hmm_vocab.txt", 'r') as f:
    voc_l = f.read().split('\n')

for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

with open("WSJ_24.pos", 'r') as f:
    y = f.readlines()

_, prep = preprocess(vocab, "test.words")

emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
states = sorted(tag_counts.keys())

accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")
predict_only("welcome",emission_counts,vocab, states)

alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)

B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
pred = viterbi_backward(best_probs, best_paths, prep, states)
m=len(pred)
print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])
print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")

find_word_pos("welcome",prep, pred)
