import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist, squareform

import os
from dotenv import load_dotenv

from google import genai
from google.genai.types import GenerationConfig, Tool

#Objective: Implement a movie recommendation system using K-means clustering.
#Authors: Fabian Fetter, Konrad Fijałkowski
#Requirements: place gemini api key in .env file as GEMINI_API_KEY, have ratings.csv in the same directory
#How to run: Run the script and input the user number when prompted.

UNSEEN_RATING = 0

"""
This script implements a movie recommendation system using K-means clustering.
It requires a Gemini API key to be placed in a .env file as GEMINI_API_KEY,
and a ratings.csv file to be present in the same directory.
Run the script and input the user number when prompted to get recommendations.
"""
def get_gemini_summary(movie_title):
    load_dotenv()
    client = genai.Client() 

    prompt = f"Provide a brief, 1-sentence plot summary for the movie {movie_title}."

    try:
        response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
        
        prompt = (
            f"for the movie '{movie_title}'. Start your response with the summary."
        )
        record = {
                'title': movie_title,
                'description': response.text.strip()
            }
        return record
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Get user index from input
"""
def get_user_index(users):

    print("Dostępni użytkownicy:")
    for i in range(len(users)):
        print(f"#{i+1} {users[i]}")
    print(f"Dla którego użytkownika chcesz uzyskać rekomendacje? (1-{len(users)})")

    while True:
        try:
            user_input = input()
            user_index = int(user_input) - 1
            if 0 <= user_index < len(users):
                selected_user = users[user_index]
                break
            else:
                print(f"Niepoprawny numer. Wprowadź liczbę od 1 do {len(users)}.")
        except ValueError:
            print("Niepoprawny format. Wprowadź numer.")
    return selected_user

"""
Extract ratings from CSV file into a nested dictionary
"""
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
    
"""
Get user rating for a specific title
"""
def get_user_rating_of_title(ratings, user, title):
    return ratings.get(user, {}).get(title, UNSEEN_RATING)

"""
Create user-item matrix from ratings data
"""
def get_user_matrix(ratings, users, all_titles):
    user_vectors = []
    for user in users:
        vector = [get_user_rating_of_title(ratings, user, title) for title in all_titles]
        user_vectors.append(vector)
    return np.array(user_vectors)

"""
Visualize clusters using PCA, by reducing dimensions to 2D
"""
def visualize_clusters(matrix, labels, users):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix)

    x = reduced[:, 0]
    y = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x, y, c=labels, s=100)

    for i, user in enumerate(users):
        plt.text(x[i] + 0.01, y[i] + 0.01, user, fontsize=9)

    plt.title("Wizualizacja klastrów użytkowników (K-means + PCA)")
    plt.xlabel("Wymiar 1 (PCA)")
    plt.ylabel("Wymiar 2 (PCA)")
    plt.colorbar(scatter, label="Klaster")
    plt.grid(True)
    plt.show()

"""
Get movie recommendations for a target user using K-Medoids clustering
"""
def get_recommendations(ratings, users, titles, target_user, number_of_clusters=5, n_recommendations=5):
    
    matrix = get_user_matrix(ratings, users, titles)
    
    pearson_distances = pdist(matrix, metric='correlation')
    distance_matrix = squareform(pearson_distances) 

    # Clustering with K-Medoids using the pre-computed distance matrix
    # KMedoids must be told the matrix is pre-computed using metric='precomputed'
    kmedoids = KMedoids(n_clusters=number_of_clusters, metric='precomputed')
    labels = kmedoids.fit_predict(distance_matrix)

    visualize_clusters(matrix, labels, users)
    
    target_user_index = users.index(target_user)
    target_cluster = labels[target_user_index]
    
    cluster_users = [users[i] for i, label in enumerate(labels) if label == target_cluster]
    
    #Get only movies that were rated by at least one user in the cluster
    cluster_matrix = matrix[[users.index(user) for user in cluster_users]]
    avg_cluster_ratings = cluster_matrix.mean(axis=0)
    user_vector = matrix[target_user_index]

    unseen_titles_indices = [i for i, rating in enumerate(user_vector) if rating == UNSEEN_RATING]

    # Remove titles with no ratings in the cluster from recommendations
    avg_cluster_ratings = np.where(cluster_matrix.sum(axis=0) == 0, -1, avg_cluster_ratings)
    
    recs = sorted(unseen_titles_indices, key=lambda i: avg_cluster_ratings[i], reverse=True)[:n_recommendations]
    unrecs = sorted(unseen_titles_indices, key=lambda i: avg_cluster_ratings[i])[:n_recommendations]

    return ([ titles[i] for i in recs ], [ titles[i] for i in unrecs ])



"""
Main execution block
"""
if __name__ == "__main__":
    load_dotenv()

    ratings = extract_ratings("ratings.csv")

    users = list(ratings.keys())

    titles = sorted({title for user_ratings in ratings.values() for title in user_ratings})

    selected_user = get_user_index(users)

    recommended_titles, unrecommended_titles = get_recommendations(ratings, users, titles, selected_user)

    recom_with_desc = []
    unrecom_with_desc = []

    for movie in recommended_titles:
        recom_with_desc.append(get_gemini_summary(movie))

    for movie in unrecommended_titles:
        unrecom_with_desc.append(get_gemini_summary(movie))

    print(f"Rekomendowane tytuły dla użytkownika {selected_user}:\n")
    for movie_record in recom_with_desc:
        print(f"\nTitle: {movie_record['title']}")
        print(f"Description: {movie_record['description']}")

    print(f"\nNierekomendowane tytuły dla użytkownika {selected_user}:")
    for movie_record in unrecom_with_desc:
        print(f"\nTitle: {movie_record['title']}")
        print(f"Description: {movie_record['description']}")