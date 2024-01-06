"""
Word search
Riia Hiironniemi 150271556
DATA.ML.100
Excersise 3
Code finds three the most similar words for input word and prints those
and distaces.
"""
import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)


# Function to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)


# Function to find the most similar words and their distances
def most_similar_words(input_word, top_n=3):
    if input_word in vocab:
        input_vector = W[vocab[input_word]]
        similarities = []
        for idx, word in enumerate(words):
            if word == input_word:
                continue
            word_vector = W[idx]
            similarity = cosine_similarity(input_vector, word_vector)
            similarities.append((word, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_words = [(input_word, 1.0)] + [(word, similarity) for word, similarity in similarities[:top_n-1]]
        return similar_words
    else:
        return []

# Main loop for analogy
while True:
    input_term = input("\nEnter a word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        similar_words = most_similar_words(input_term)
        if similar_words:
            print("\n{:>20} {:>20}".format("Word", "Distance"))
            print("-" * 40)
            for word, similarity in similar_words:
                print("{:>20} {:>20}".format(word, similarity))
        else:
            print(f"'{input_term}' not found in the vocabulary.")
