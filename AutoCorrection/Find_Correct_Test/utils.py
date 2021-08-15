import pickle

def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''
    delete_l = []
    split_l = []

    split_l = [(word[:i],word[i:]) for i in range(len(word)+1)]
    delete_l = [L+R[1:] for L,R in split_l if R]

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")
    return delete_l

def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    switch_l = []
    split_l = []

    split_l = [(word[:i],word[i:]) for i in range(len(word)+1)]
    switch_l = [L[:-1]+R[0]+L[-1]+R[1:] for L,R in split_l if R and L]

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")
    return switch_l

def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_set = []
    split_l = []

    split_l = [(word[:i],word[i:]) for i in range(len(word)+1)]
    replace_set = [ L[:-1] + c + R for L,R in split_l if L for c in letters if c != L[-1]]
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")
    return replace_l

def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    split_l = [(word[:i],word[i:]) for i in range(len(word)+1)]
    insert_l = [ L + c + R for L,R in split_l for c in letters]

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")
    return insert_l

def edit_one_letter(word, allow_switches = True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()
    f1 = []
    f2 = []
    f3 = []
    f4 = []

    f1 = insert_letter(word, verbose=False)
    f2 = replace_letter(word, verbose=False)
    f3 = delete_letter(word, verbose=False)
    if allow_switches:
        f4 = switch_letter(word, verbose=False)
    edit_one_set = set(f1+f2+f3+f4)

    return edit_one_set


def edit_two_letters(word, allow_switches = True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()

    oneL = edit_one_letter(word, allow_switches = True)
    for i in oneL:
        edit_two_set = set(edit_one_letter(i, allow_switches = True)) | edit_two_set

    return edit_two_set


def get_corrections(word, probs, vocab, n=2, verbose = False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = []
    n_best = []

    set_1 = edit_one_letter(word, allow_switches = True)
    set_2 = edit_two_letters(word, allow_switches = True)
    if word in vocab:
        suggestions.append((word,probs[word]))
    else:
        suggestions = [(i,probs[i]) for i in set_1 if i in vocab] or [(i,probs[i]) for i in set_2 if i in vocab] or [(word,0)]
    n_best = sorted(suggestions, key = lambda x : x[-1], reverse = True)[:n]

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)
    return n_best

def load_files():
    with open('probs.pickle', 'rb') as handle:
        probs = pickle.load(handle)
    with open('probs2.pickle', 'rb') as handle:
        probs2 = pickle.load(handle)
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    with open('vocab2.pickle', 'rb') as handle:
        vocab2 = pickle.load(handle)
    return probs, vocab, probs2, vocab2
