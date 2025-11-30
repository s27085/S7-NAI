#Objective: Implement decision tree and support vector methods clustering algorithms on a sonar readings dataset
#Authors: Fabian Fetter, Konrad Fijałkowski
#Requirements: run with CLI from the same directory where data.csv is present

import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import tree
from sklearn import svm
import numpy as np
from sklearn.calibration import LabelEncoder

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
    
    plt.figure(figsize=(25, 10))
    tree.plot_tree(prediction, 
          filled=True,
          class_names=['Mine', 'Rock'],
          rounded=True,
          fontsize=10)
    
    plt.show()

def visualize_svm_decision_boundary(clf, sample_features, class_labels):

    coefs = np.abs(clf.coef_[0])
    top_2_indices = np.argsort(coefs)[-2:][::-1] 
    i, j = top_2_indices[0], top_2_indices[1]

    mean_all_features = sample_features.mean().values
    
    le = LabelEncoder()
    y_numeric = le.fit_transform(class_labels)
    PADDING = 0.05 
    x_min, x_max = sample_features.iloc[:, i].min() - PADDING, sample_features.iloc[:, i].max() + PADDING
    y_min, y_max = sample_features.iloc[:, j].min() - PADDING, sample_features.iloc[:, j].max() + PADDING
    step_size = 0.01 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    grid_points_2D = np.c_[xx.ravel(), yy.ravel()]
    N_grid_points = grid_points_2D.shape[0]
    X_predict = np.tile(mean_all_features, (N_grid_points, 1))
    X_predict[:, i] = grid_points_2D[:, 0] 
    X_predict[:, j] = grid_points_2D[:, 1] 
    
    Z_string = clf.predict(X_predict) 
    Z_numeric = le.transform(Z_string)
    Z = Z_numeric.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    plt.scatter(
        sample_features.iloc[:, i], 
        sample_features.iloc[:, j], 
        c=y_numeric, 
        cmap=plt.cm.coolwarm, 
        edgecolors='k', 
        s=40
    )
    
    plt.title(f"SVM (linear kernel) Decision Boundary (Slice at Mean: Features {i} vs {j})")
    plt.xlabel(f"Feature {i}")
    plt.ylabel(f"Feature {j}")
    plt.show()

def SVM(sample_features, class_labels, visualize_tree, kernel_func='linear'):
    clf = svm.SVC(probability=True, kernel=kernel_func)
    clf.fit(sample_features, class_labels)

    single_row = sample_features.iloc[[117]]
    result = clf.predict_proba(single_row)
    print(f"SVM ({kernel_func} kernel) Predicted probabilities for first sample row: Mine: {result[0][MINE]*100:.2f}%, Rock: {result[0][ROCK]*100:.2f}%")
    if kernel_func in ['linear']:
        visualize_svm_decision_boundary(clf, sample_features, class_labels) if visualize_tree else None


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

