####################################################################### 
#
# An example of Item-Based Collaborative Filtering.
#
# We have books, users and ratings. If a user has not provided
# a rating for a book, we assume that the user has not read that 
# book.
# 
# Perform Cosine Similarity on books to find similar books. Then,
# to recommend a book to Bob, find books that Bob has read and
# has given high ratings. Next, find books that Bob has not read but 
# has high similarity scores against those books that Bob has given
# high ratings to. Recommed those books to Bob.
#
#####################################################################

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# hashmap for books (bk-id, bk-name)
books = {
    "b1": "Code Breakers",
    "b2": "Shadow Cipher",
    "b3": "Binary Intrigues",
    "b4": "Zero Day",
    "b5": "Whispers Protocol",
    "b6": "Lunar Serenade",
    "b7": "Virtual Networks",
    "b8": "Port Knocking",
    "b9": "Code Warriors",
    "b10": "Byte Bandits"
}

# hashmap for users (uid, username)
users = {
    "u1": "John",
    "u2": "Mary",
    "u3": "Bob",
    "u4": "James",
    "u5": "Larry",
    "u6": "Tom"
}

# items-ratings by users (ratings from 1 to 5).
# a rating of 0 means item is not rated.
# indexes represent "books", columns represent "users"
ratings = np.array([
    [2, 3, 5, 4, 4, 0],
    [4, 4, 3, 0, 0, 4],
    [4, 5, 5, 0, 4, 0],
    [4, 4, 5, 2, 4, 1],
    [3, 5, 0, 4, 2, 0],
    [0, 4, 2, 4, 3, 3],
    [3, 5, 4, 2, 0, 2],
    [4, 0, 4, 3, 5, 4],
    [5, 0, 4, 4, 4, 0],
    [3, 5, 4, 5, 0, 0],
])

# dataframe to show books-users matrix, with ratings as values
df_rates = pd.DataFrame(ratings, 
        index=books.keys(),   # (e.g. "b1", "b2")
        columns=users.keys()) # (e.g. "u1", "u2")
print(f"\nMatrix of user-ratings on books ...\n{df_rates}")

# find similar items (based on users' ratings)
sim_mat = cosine_similarity(df_rates)
df_sim = pd.DataFrame(sim_mat,
        index=books.keys(),
        columns=books.keys())
print(f"\nSimilarity based on user-ratings ...\n{df_sim}")

# recommend books to a user (pick any user)
target_uid = "u6"

# get books rated highly by the user
limit_rate = 4
user_rates = df_rates.loc[:,target_uid]
high_rates = user_rates[user_rates >= limit_rate]
print(f"\nBooks rated highly by \"{target_uid}\" ...\n{high_rates}")

# sort user's ratings in ascending order
user_rates = user_rates.sort_values(ascending=False)

# get books not read by user (denoted as no ratings from user) 
no_rates = user_rates[user_rates == 0]
print(f"\nBooks that user \"{target_uid}\" has not read/rated ...\n{no_rates}")

# extract similarity scores of books that are highly rated by user, 
# but not yet read by user.
df_targets = df_sim.loc[high_rates.index, no_rates.index]
print(f"\nBooks that user \"{target_uid}\" likes vs books he has not read ...\n{df_targets}")

# get top 3 books with highest similarity scores to recommend
df_mean = df_targets.mean(axis=0).sort_values(ascending=False)[0:3]
print(f"\nRecommended books for \"{target_uid}\" =\n{df_mean}")

# display book-names
print("\nRecommended book titles:")
for book_id in df_mean.index:
  print(f"  {books[book_id]}")
