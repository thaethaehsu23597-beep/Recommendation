import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


############################################################ 
# Function to list all word-embeddings in our vocab.
############################################################
def list_word_embeddings(model):
    vocab = list(model.wv.key_to_index.keys())
    for word in vocab:
        print(f"{word}: {model.wv[word]}")


############################################################ 
# Function to get the average vector for a sentence
############################################################
def get_sentence_vector(sentence, model):
    word_vectors = [model.wv[word] for word in sentence if word in model.wv]

    # take the average of all word vectors in the sentence
    return np.mean(word_vectors, axis=0)


#####################################################
# Preprocess text data
#####################################################
def preprocess(text):
    # convert to lowercase
    text = text.lower()

    # remove punctuation
    trans = str.maketrans("", "", string.punctuation)
    text = text.translate(trans)

    # remove stop-words
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # stemming
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


#####################################################
# Main
#####################################################
quotes = [
    "Let your life be shaped by decisions you made, not by the ones you didn't.",
    "Our greatest fear should not be of failure, but of succeeding at things in life that don't really matter.",
    "You can live your whole life and never know who you are; until you see the world through the eyes of others.",
    "The eyes of others our prisons, their thoughts our cages.",
    "The mark of a successful man is one that has spent an entire day on the bank of a river without feeling guilty about it.",
    "In the beginner's mind, there are many possibilities; in the expert's mind, there are few.",
    "You attract into your life that which you are.",
    "We all make decisions, but in the end, our decisions made us."
]

# Preprocess text
sentences = []
for text in quotes:
    sentences.append(preprocess(text))
   
# Train Word2Vec model.
# "window=2" means context window size is 2 words before and after the target word.
# "min_count=1" means even words that appear just once in the corpus are included in the training.
# "sg=0" means CBOW model.
model = Word2Vec(sentences, vector_size=5, window=2, min_count=1, sg=0)

# List the trained word-embeddings
print(f"\nWord-Embeddings:")
list_word_embeddings(model)

# Get embeddings for each sentence
sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in sentences])

# Compute cosine similarity among sentence-embeddings
similarity_matrix = cosine_similarity(sentence_vectors)

# Display the cosine similarity matrix among the sentences
headers = ["sentence " + str(i+1) for i in range(len(quotes))]
df = pd.DataFrame(similarity_matrix,
             index=headers,
             columns=headers)
print(f"\nCosine Similarity:\n{df}")
