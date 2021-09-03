# NLP_Repo

Projects for NLP specialization on Coursera.

* Auto Complete - Building a language model using n-gram probability to suggest the next word/words for a given sentence

* Auto Correction - Find misspelled word and correct it by doing the next steps:

	a- identify misspelled word
	
	b- find words "n-edits" away from the misspelled word
	
	c- calculates the probability of the word
	
  * The program doesn't use neural networks but depends on the corpus for probs
	

* Language Model GRU - Improvement to the auto complete project by using gated recurrent unit (GRU) to build a language model insted of n-gram method.
	* The model is built using Trax framework from Google

* Machine Translation KNN - Simple english to french machine translator using word2vec dataset to encode the sentence and k-nearest neighbors (KNN) to find the most probable sentence

* Named Entity Recognition LSTM - NER using Embedded layer and LSTM network on Trax

* Neural Machine Translation with Attention - English to German Translator using Attention Model and LSTM

* Parts-of-Speech Tagging - using Viterbi algorithm and hidden markov model (HMM) to determine the <PoS> of a given word
	
* Question Duplicates - Detect if two questions are duplicate or not, using Trax built a siamese network with LSTM and triplet loss


* Sentiment Analysis on Tweets:
	* Version 1 - Using Logistic Regression
	* Version 2 - Using Naive Bayes
	* Version 3 - Using Deep Neural Network on Trax
