from utils import load_files, get_corrections

probs, vocab, probs2, vocab2 = load_files()
my_word = 'wrod' 
tmp_corrections = get_corrections(my_word, probs2, vocab2, 2, verbose=True)
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")
