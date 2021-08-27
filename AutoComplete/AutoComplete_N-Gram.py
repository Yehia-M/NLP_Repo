#!/usr/bin/env python

import math
import random
import numpy as np
import pandas as pd
import nltk

def split_to_sentences(data):
    """
    Split data by linebreak "\n"
    Args:
        data: str

    Returns:
        A list of sentences
    """

    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences

def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences

def get_tokenized_data(data):
    """
    Make a list of tokenized sentences

    Args:
        data: String

    Returns:
        List of lists of tokens
    """
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token,0) + 1

    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more

    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.

    Returns:
        List of words that appear N times or more
    """
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)
    for word, cnt in word_counts.items():
        if cnt >= count_threshold:
            closed_vocab.append(word)
    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.

    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words

    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)

        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token="<unk>")
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token="<unk>")
    return train_data_replaced, test_data_replaced, vocabulary


def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: number of words in a sequence

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    n_grams = {}
    for sentence in data:
        sentence = ['<s>']*n + sentence + ['<e>']
        sentence = tuple(sentence)

        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i+n]
            n_grams[n_gram] = n_grams.get(n_gram,0) + 1

    return n_grams


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing

    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter

    Returns:
        A probability
    """
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram,0)
    denominator = previous_n_gram_count + k*vocabulary_size
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)

    numerator = n_plus1_gram_count + k

    probability = numerator/denominator

    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing

    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to the probability.
    """

    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    Get suggestion for the next word

    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length > n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          - string of the most likely next word
          - corresponding probability
    """

    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    suggestion = None
    max_prob = 0

    for word, prob in probabilities.items():
        # If the optional start_with string is set
        if start_with:
            if not word.startswith(start_with):
                continue

        if prob > max_prob:
            suggestion = word
            max_prob = prob

    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions


with open("en_US.twitter.txt", "r") as f:
    data = f.read()

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]


minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data,test_data,minimum_freq)

n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)

previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest4)


previous_tokens = ["i", "want", "to", "go"]
tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest5)

previous_tokens = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest6)

previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest7)


with open("big.txt", "r") as f:
    data2 = f.read()
print("Data type:", type(data2))
print("Number of letters:", len(data2))

tokenized_data2 = get_tokenized_data(data2)
random.seed(87)
random.shuffle(tokenized_data2)

train_size2 = int(len(tokenized_data2) * 0.8)
train_data2 = tokenized_data[0:train_size2]
test_data2 = tokenized_data[train_size2:]
print(len(train_data2))

minimum_freq = 2
train_data_processed2, test_data_processed2, vocabulary2 = preprocess_data(train_data2,test_data2,minimum_freq)

n_gram_counts_list2 = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts2 = count_n_grams(train_data_processed2, n)
    n_gram_counts_list2.append(n_model_counts2)


previous_tokens2 = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens2, n_gram_counts_list2, vocabulary2, k=1.0)

print(f"The previous words are {previous_tokens2}, the suggestions are:")
display(tmp_suggest7)

previous_tokens2 = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens2, n_gram_counts_list2, vocabulary2, k=1.0)

print(f"The previous words are {previous_tokens2}, the suggestions are:")
display(tmp_suggest6)

previous_tokens2 = ["i", "want", "to", "go"]
tmp_suggest6 = get_suggestions(previous_tokens2, n_gram_counts_list2, vocabulary2, k=1.0)

print(f"The previous words are {previous_tokens2}, the suggestions are:")
display(tmp_suggest6)

previous_tokens2 = ["i", "am", "to"]
tmp_suggest6 = get_suggestions(previous_tokens2, n_gram_counts_list2, vocabulary2, k=1.0)

print(f"The previous words are {previous_tokens2}, the suggestions are:")
display(tmp_suggest6)
