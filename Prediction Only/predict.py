import numpy as np
import pickle
from utils_Final import process_tweet, build_freqs, extract_features,sigmoid,predict_tweet

with open('freqs.pickle', 'rb') as handle:
    freqs = pickle.load(handle)

theta = np.array([[7.25177398e-08], [5.23910434e-04],[-5.55171581e-04]])

my_tweet = "Only 10 minutes and Ancelotti's hand is already very noticeable. I hope I get the best out of players like Bale, Isco, Jovic and Odegaard. I'm excited about this season."
predict_tweet(my_tweet, freqs, theta)
