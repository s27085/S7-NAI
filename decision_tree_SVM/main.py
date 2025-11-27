#Objective: Implement decision tree and support vector methods clustering algorithms on a sonar readings dataset
#Authors: Fabian Fetter, Konrad Fijałkowski
#Requirements: run with CLI from the same directory where data.csv is present

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

ROCK = 0
MINE = 1
DATA_FILENAME = 'data.csv'

def read_sonar_data():
    data_file = pd.read_csv(filepath_or_buffer=DATA_FILENAME, decimal=".", delimiter=",")
    features = data_file.iloc[:, :-1] #read everything except last column
    labels = data_file.iloc[:, -1] # read only the last column
    return features, labels

def plot_decision_tree(prediction):
    plt.figure(figsize=(20, 10))

    tree.plot_tree(prediction, 
          filled=True,
          class_names=['Mine', 'Rock'],
          rounded=True,
          fontsize=10)
    
    plt.show()

def decision_tree():
    sample_features, class_labels = read_sonar_data()
    clf = tree.DecisionTreeClassifier()
    clf.fit(sample_features, class_labels)

    single_row = sample_features.iloc[[0]]
    
    result = clf.predict_proba(single_row)
    plot_decision_tree(clf)

if __name__ == "__main__":
    decision_tree()