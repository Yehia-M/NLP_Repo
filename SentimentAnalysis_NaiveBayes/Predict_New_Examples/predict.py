import pickle
from utils import process_tweet, lookup, naive_bayes_predict

#Load parameters
with open('loglikelihood.pickle', 'rb') as handle:
    loglikelihood = pickle.load(handle)
with open('logprior.pickle', 'rb') as handle:
    logprior = pickle.load(handle)

tweet = "Happy day and nice weather :)"
p = naive_bayes_predict(tweet, logprior, loglikelihood)
print(p)
