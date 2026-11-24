import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#q1
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

ratings_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)



ratings_matrix = ratings_matrix.fillna(0)

print(ratings_matrix.shape)

# Q1(a)
movie_counts = ratings.groupby("movieId").size().reset_index(name="count")

top_movies = movie_counts.sort_values("count", ascending=False).head(3)

top_movies = top_movies.merge(movies, on="movieId")

print(top_movies[["title", "count"]])

# Q1(b)
user_counts = ratings.groupby("userId").size().reset_index(name="count")

top_users = user_counts.sort_values("count", ascending=False).head(3)

print(top_users)



# Q2(a)
k_values = [2, 4, 8, 16, 32, 64, 128]
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(ratings_matrix)
    inertias.append(kmeans.inertia_)

print(inertias)

plt.plot(range(len(k_values)), inertias, marker="o")
plt.xticks(range(len(k_values)), k_values)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Inertia")
plt.show()


# Q2(c)
# Q2(c)
kmeans = KMeans(n_clusters=8, random_state=0)
clusters = kmeans.fit_predict(ratings_matrix)

print(np.bincount(clusters))

for cluster in range(8):
    cluster_data = ratings_matrix[clusters == cluster]

    minimum_ratings = len(cluster_data) * 0.10

    rating_counts = (cluster_data > 0).sum()
    valid_movies = rating_counts[rating_counts >= minimum_ratings].index

    average_ratings = cluster_data[valid_movies].replace(0, np.nan).mean()
    top_movies = average_ratings.sort_values(ascending=False).head(3)

    print("\nCluster", cluster)

    for movie_id, rating in top_movies.items():
        title = movies[movies["movieId"] == movie_id]["title"].iloc[0]
        print(title, round(rating, 2))