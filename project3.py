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


# Q3(a)
movie_matrix = ratings_matrix.T

mean_centered = movie_matrix - movie_matrix.mean()

print(mean_centered.shape)

# Q3(b)
pca = PCA(n_components=2)

pca_data = pca.fit_transform(mean_centered)

print(pca_data.shape)


# Q3(c)
movie_info = movies[movies["movieId"].isin(ratings_matrix.columns)].copy()

movie_info["first_genre"] = movie_info["genres"].str.split("|").str[0]

movie_info["PCA1"] = pca_data[:, 0]
movie_info["PCA2"] = pca_data[:, 1]

for genre in movie_info["first_genre"].unique():
    genre_data = movie_info[movie_info["first_genre"] == genre]

    plt.scatter(
        genre_data["PCA1"],
        genre_data["PCA2"],
        label=genre,
        s=10
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Movies by Genre")
plt.legend()
plt.show()


# Q3(d)
pca_full = PCA()
pca_full.fit(mean_centered)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

components_80 = np.argmax(cumulative_variance >= 0.80) + 1
components_40 = np.argmax(cumulative_variance >= 0.40) + 1

print("80%:", components_80)
print("40%:", components_40)


# Q4(a)
svd = TruncatedSVD(n_components=128, random_state=0)
svd.fit(ratings_matrix)

plt.plot(svd.singular_values_)
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.title("SVD Singular Values")
plt.show()


# Q4(b)
svd_k_values = [2, 4, 8, 16, 32, 64, 128]

for k in svd_k_values:
    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit(ratings_matrix)

    explained_variance = svd.explained_variance_ratio_.sum()

    print(k, explained_variance)


# Q4(c)
svd_2 = TruncatedSVD(n_components=2, random_state=0)

svd_data = svd_2.fit_transform(ratings_matrix)

print(svd_data.shape)

# Q4(d)
for cluster in range(8):
    cluster_data = svd_data[clusters == cluster]

    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label="Cluster " + str(cluster),
        s=20
    )

plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVD of Users by Cluster")
plt.legend()
plt.show()