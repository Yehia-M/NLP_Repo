import pickle

with open('pred.pickle', 'rb') as handle:
    pred = pickle.load(handle)

with open('prep.pickle', 'rb') as handle:
    prep = pickle.load(handle)

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

word = 'welcome'
print(find_word_pos(word,prep, pred))
