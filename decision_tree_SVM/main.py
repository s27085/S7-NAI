#Objective: Implement decision tree and support vector methods clustering algorithms on a sonar readings dataset
#Authors: Fabian Fetter, Konrad Fijałkowski
#Requirements: run with CLI from the same directory where data.csv is present

import pandas as pd
from sklearn import tree

ROCK = 0
MINE = 1
DATA_FILENAME = 'data.csv'

def read_sonar_data(filename):
    data_file = pd.read_csv(delimiter=",", filepath_or_buffer="data.csv", decimal=".")
    features = data_file.iloc[:, :-1] #read everything except last column
    labels = data_file.iloc[:, -1] # read only the last column
    return features, labels

def decision_tree():
    sample_features, class_labels = read_sonar_data(DATA_FILENAME)
    clf = tree.DecisionTreeClassifier()
    clf.fit(sample_features, class_labels)

    single_row = sample_features.iloc[[0]]
    
    result = clf.predict_proba(single_row)
    print(result)

if __name__ == "__main__":
    decision_tree()