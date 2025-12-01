import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# any ratings equal or highly means user likes
like_threshold = 4.0    

# recommending books for this user
user = "larry"

df = pd.read_csv("mini_item_colab.csv", index_col=0)

# mean-centering (normalize rating bias)
users_mean = np.mean(df, axis=1)
df_mean_cent = df.sub(users_mean, axis=0)

# item-based, so items are rows and ratings are columns
items_df = df_mean_cent.T  
print(f"items_df=\n{items_df}\n")

# leave out target user (calculate item-similarities based on other users)
other_users_df = items_df.drop(columns=user, axis=0)  # drop target user's ratings

# compute similarity among items
sim_mat = cosine_similarity(other_users_df)
sim_df = pd.DataFrame(data=sim_mat,
                      index=items_df.index,
                      columns=items_df.index)
print(f"sim_df=\n{sim_df}\n")

# books that Larry likes
cond = df.loc[user] >= like_threshold
user_likes = df.loc[user][cond].index
user_not_read = sim_df.drop(columns=user_likes).columns

# find highest cosine similiarity values
x = sim_df.loc[user_likes, user_not_read]
top = x.idxmax(axis=1)

print("Recommended Books:", np.unique(top.values))






