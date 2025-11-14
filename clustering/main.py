import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Objective: Implement a movie recommendation system using K-means clustering.
#Authors: Fabian Fetter, Konrad Fijałkowski
#How to run: Ensure you have 'ratings.csv' in the same directory. Run the script and input the user number when prompted.


def extract_ratings(filename):
    try:
        if filename == "":
            raise ValueError
        datafile = pd.read_csv(filename, header=None)
        records_list = datafile.to_dict("records")
        final_dict = {}
        for record in records_list:
            person_id = record.get(0) 
            
            if person_id:
                ratings_data = {}
                for i in range(1, len(record), 2):
                    if i+1 in record:
                        try:
                            ratings_data[str(record[i])] = int(record[i+1])
                        except (ValueError, TypeError):
                            continue
                
                final_dict[person_id] = ratings_data
        return final_dict
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except pd.errors.EmptyDataError:
        print(f"The file {filename} is empty.")
        return []
    except ValueError:
        print(f"Filename was not specified")
        return []
    

def get_user_matrix(ratings, users, all_titles):
    user_vectors = []
    for user in users:
        vector = [ratings[user].get(title, 0) for title in all_titles]
        user_vectors.append(vector)
    return np.array(user_vectors)

def visualize_clusters(matrix, labels, users):
    # Redukujemy wymiar z N filmów -> 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix)

    x = reduced[:, 0]
    y = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x, y, c=labels, s=100)

    # Opisujemy punkty nazwami użytkowników
    for i, user in enumerate(users):
        plt.text(x[i] + 0.01, y[i] + 0.01, user, fontsize=9)

    plt.title("Wizualizacja klastrów użytkowników (K-means + PCA)")
    plt.xlabel("Wymiar 1 (PCA)")
    plt.ylabel("Wymiar 2 (PCA)")
    plt.colorbar(scatter, label="Klaster")
    plt.grid(True)
    plt.show()


def get_recommendations_kmeans(ratings, users, titles, target_user, k=3, n_recommendations=5):
    
    kmeans = KMeans(n_clusters=k, random_state=69, n_init=10)
    matrix = get_user_matrix(ratings, users, titles)
    labels = kmeans.fit_predict(matrix)
    
    target_index = users.index(target_user)
    target_cluster = labels[target_index]
    
    cluster_users = [users[i] for i, label in enumerate(labels) if label == target_cluster]
    
    cluster_matrix = matrix[[users.index(user) for user in cluster_users]]
    avg_cluster_ratings = cluster_matrix.mean(axis=0)
    user_vector = matrix[target_index]

    unseen_titles_indices = [i for i, rating in enumerate(user_vector) if rating == 0]

    # Odrzucamy filmy, które nie były oceniane przez nikogo w klastrze
    avg_cluster_ratings = np.where(cluster_matrix.sum(axis=0) == 0, -1, avg_cluster_ratings)
    
    recs = sorted(unseen_titles_indices, key=lambda i: avg_cluster_ratings[i], reverse=True)[:n_recommendations]
    unrecs = sorted(unseen_titles_indices, key=lambda i: avg_cluster_ratings[i])[:n_recommendations]

    visualize_clusters(matrix, labels, users)
    return ([ titles[i] for i in recs ], [ titles[i] for i in unrecs ])

ratings = extract_ratings("ratings.csv")

users = list(ratings.keys())

titles = sorted({title for user_ratings in ratings.values() for title in user_ratings})

for i in range(len(users)):
    print(f"#{i+1} {users[i]}")
print(f"Dla którego użytkownika chcesz uzyskać rekomendacje? (1-{len(users)})")
user_index = int(input()) - 1

recommended_titles, unrecommended_titles = get_recommendations_kmeans(ratings, users, titles, users[user_index])

print(f"Rekomendowane tytuły dla użytkownika {users[user_index]}:")
for title in recommended_titles:
    print(f"- {title}")

print(f"Nierekomendowane tytuły dla użytkownika {users[user_index]}:")
for title in unrecommended_titles:
    print(f"- {title}")