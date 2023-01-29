import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import networkx as nx
import community

# load the dataframe
df = pd.read_csv('D:\\Code\\MRS\\user_ratings.csv')

# preprocessing
# normalize the ratings and anxiety scores by substracting the minimum value to each score and divide by the range.
# Another option would be Standardization, which substracts the mean to the value, and divides by standard deviation.
scaler = MinMaxScaler()
df[['rating', 'anxiety_score']] = scaler.fit_transform(df[['rating', 'anxiety_score']])

# create a user-song matrix
user_song_matrix = df.pivot_table(values='rating', index='user_id', columns='song_id').fillna(0)

# split the data into train and test sets
train, test = train_test_split(user_song_matrix, test_size=0.2)

# perform the NMF with ANLS
def nmf_anls(X, k=10, max_iter=50):
    m, n = X.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)
    for i in range(max_iter):
        H = np.linalg.solve(W.T @ W, W.T @ X)
        W = np.linalg.solve(H @ H.T, H @ X.T)
        X_pred = W @ H
        error = mean_absolute_error(X, X_pred)
        print(f'Iteration {i}, Error: {error}')
    return W, H

W, H = nmf_anls(train, k=10)

# use the latent feature matrices to recommend songs
def recommend_songs(X, user_idx, n_songs=5):
    user_latent_features = W[user_idx]
    song_latent_features = H.T
    similarity = cosine_similarity(user_latent_features.reshape(1, -1), song_latent_features)
    top_songs = np.argsort(-similarity, axis=1)[:, :n_songs]
    return top_songs

user_idx = 0
top_songs = recommend_songs(train, user_idx)

# create a similarity graph with the cosine similarity of the latent features
song_similarity = cosine_similarity(H.T)
G = nx.Graph(song_similarity)

# use the Louvain method to find communities in the graph
partition = community.best_partition(G)

# print the communities
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]

import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = [partition[n] for n in G.nodes()])
nx.draw_networkx_edges(G, pos, edge_color='r', arrows=False)
plt.show()
