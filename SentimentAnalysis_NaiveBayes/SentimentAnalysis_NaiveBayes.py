#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd


# In[2]:


# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


# In[3]:


def count_tweets(result, tweets, ys):
    for tweet,label in zip(tweets,ys):
        cleaned_TWT = process_tweet(tweet) 
        for word in cleaned_TWT:
            result[(word,label)] = result.get((word,label),0) + 1
    return result    


# In[4]:


# Build the freqs dictionary
freqs = count_tweets({}, train_x, train_y)


# In[5]:


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0
    
    UniqueV = set([i[0] for i in freqs.keys()])
    V = len(UniqueV)
    
    Npos = 0
    Nneg = 0
    for key,value in freqs.items():
        if key[1] == 1:
            Npos = Npos + value
        else:
            Nneg = Nneg + value
            
    Dpos = np.sum(train_y == 1)
    Dneg = np.sum(train_y == 0)
    logprior = np.log(Dpos/Dneg)
    
    for word in UniqueV:
        fpos = freqs.get((word,1),0)
        fneg = freqs.get((word,0),0)
        
        Ppos = (fpos + 1) / (Npos + V)
        Pneg = (fneg + 1) / (Nneg + V)
        
        loglikelihood[word] = np.log(Ppos/Pneg)
   
    return logprior, loglikelihood


# In[6]:


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


# In[7]:


def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    prediction = logprior
    for word in process_tweet(tweet):
        if word in loglikelihood:
            prediction += loglikelihood[word]
        
    return prediction


# In[8]:


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    
    # error is the average of the absolute values of the differences between y_hats and test_y
    error = sum(y_hats != test_y)
    error = error/len(test_x)

    accuracy = 1-error
    return accuracy


# In[9]:


print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))


# In[10]:


for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 
              'great great great great','Today is a happy day']:
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
    print(f'{tweet} -> {p:.2f}')

