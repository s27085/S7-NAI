#Objective: Implement decision tree and support vector methods clustering algorithms on a sonar readings dataset
#Authors: Fabian Fetter, Konrad Fijałkowski
#Requirements: run with CLI from the same directory where data.csv is present

import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import tree
from sklearn import svm

MINE = 0
ROCK = 1
DATA_FILENAME = 'data.csv'
VISUALIZE_TREES = 1

def read_sonar_data():
    data_file = pd.read_csv(filepath_or_buffer=DATA_FILENAME, decimal=".", delimiter=",")
    features = data_file.iloc[:, :-1] #read everything except last column
    labels = data_file.iloc[:, -1] # read only the last column
    return features, labels


def visualize_decision_tree(prediction):
    
    plt.figure(figsize=(20, 10))
    tree.plot_tree(prediction, 
          filled=True,
          class_names=['Mine', 'Rock'],
          rounded=True,
          fontsize=10)
    
    plt.show()

def SVM(sample_features, class_labels, visualize_tree, kernel_func='linear'):
    clf = svm.SVC(probability=True, kernel=kernel_func)
    clf.fit(sample_features, class_labels)

    single_row = sample_features.iloc[[117]]
    result = clf.predict_proba(single_row)
    print(f"SVM ({kernel_func} kernel) Predicted probabilities for first sample row: Mine: {result[0][MINE]*100:.2f}%, Rock: {result[0][ROCK]*100:.2f}%")


def decision_tree(sample_features, class_labels, visualize_tree):
    clf = tree.DecisionTreeClassifier()
    clf.fit(sample_features, class_labels)
    visualize_decision_tree(clf) if visualize_tree else None
    single_row = sample_features.iloc[[117]]
    
    result = clf.predict_proba(single_row)
    print(f"DT Predicted probabilities for first sample row: Mine: {result[0][MINE]*100:.2f}%, Rock: {result[0][ROCK]*100:.2f}%")

if __name__ == "__main__":
    sample_features, class_labels = read_sonar_data()

    decision_tree(sample_features, class_labels, visualize_tree=VISUALIZE_TREES)
    SVM(sample_features, class_labels, visualize_tree=VISUALIZE_TREES, kernel_func='linear')
    SVM(sample_features, class_labels, visualize_tree=VISUALIZE_TREES, kernel_func='rbf')
    SVM(sample_features, class_labels, visualize_tree=VISUALIZE_TREES, kernel_func='poly')
    SVM(sample_features, class_labels, visualize_tree=VISUALIZE_TREES, kernel_func='sigmoid')

