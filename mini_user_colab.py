import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# two users must at least have this cosine similarity
# values to be deemed as similar users
sim_threshold = 0.45     

# any ratings equal or highly means user like
like_threshold = 4.0    

# recommend books for this user
user = "larry"  

# index_col=0 means column 0 will be used as the index for our dataframe
df = pd.read_csv("mini_user_colab.csv", index_col=0)

# mean-centering (normalize rating bias)
users_mean = np.mean(df, axis=1)
df_mean_cent = df.sub(users_mean, axis=0)

# consider books that user has already read
user_row = df.loc[user]
bks_user_read = user_row[user_row.notna()].index
sub_df = df_mean_cent[bks_user_read]

# compute similarity among users
sim_mat = cosine_similarity(sub_df)
sim_df = pd.DataFrame(data=sim_mat,
                      index=df.index,
                      columns=df.index)
print(f"sim_df=\n{sim_df}\n")

# locate similar users
sim_df.drop(columns=user, inplace=True)
user_sim_row = sim_df.loc[user]
sim_users = user_sim_row[user_sim_row >= sim_threshold].index
print(sim_users)

# locate books that similar users have given high ratings
sim_users_ratings = df.loc[sim_users].drop(columns=bks_user_read)
print(sim_users_ratings)

col_mask = (sim_users_ratings >= like_threshold).any(axis=0)
bks_to_recommend = sim_users_ratings.columns[col_mask].tolist()
print("Recommended Books:", bks_to_recommend)
