"""
This script implements decision tree and support vector machine (SVM) algorithms for clustering sonar readings data. 
It includes functionalities for data visualization, algorithm metrics evaluation, and prediction analysis.

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Objective:
- Implement decision tree and SVM algorithms on a sonar readings dataset.

Requirements:
- The script should be run from the command line in the same directory where the `data.csv

Constants:
- MINE, ROCK: Class labels for the dataset.
- DATA_FILENAME: Name of the CSV file containing the dataset.
- VISUALIZE_TREES, PRINT_REPORT: Flags for visualization and report printing.
- SAMPLE_ROW_INDEX: Index of the sample row for prediction analysis.

Usage:
Run the script from the command line in the same directory as the `data.csv` file. 

Other example of data:
https://raw.githubusercontent.com/MachineLearningBCAM/Datasets/refs/heads/main/data/multi_class_datasets/vehicle.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn import tree
from sklearn import svm
import numpy as np
from sklearn.calibration import LabelEncoder

MINE = 0
ROCK = 1
DATA_FILENAME = 'data.csv'
VISUALIZE_TREES = 0
PRINT_REPORT = 0
SAMPLE_ROW_INDEX = 11

def read_sonar_data():
    """
    Reads, shuffles, encodes, and splits sonar data manually.
    Returns: x_train, x_test, y_train, y_test, class_names
    """
    df = pd.read_csv(DATA_FILENAME, header=None)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y_column = df.iloc[:, -1]
    unique_classes = sorted(y_column.unique())
    label_map = {name: i for i, name in enumerate(unique_classes)}
    df.iloc[:, -1] = df.iloc[:, -1].map(label_map)

    all_x = df.iloc[:, :-1].values.astype("float32")
    all_y = df.iloc[:, -1].values.astype("int32")

    num_samples = len(df)
    train_size = int(0.8 * num_samples)

    x_train = all_x[:train_size]
    y_train = all_y[:train_size]

    x_test = all_x[train_size:]
    y_test = all_y[train_size:]
    
    return x_train, x_test, y_train, y_test, unique_classes


def visualize_decision_tree(prediction):
    """
    Visualizes a decision tree classifier.

    Args:
        prediction (DecisionTreeClassifier): The trained decision tree classifier to visualize.
    """
    
    plt.figure(figsize=(25, 10))
    tree.plot_tree(prediction, 
          filled=True,
          class_names=['Mine', 'Rock'],
          rounded=True,
          fontsize=10)
    
    plt.show()

def visualize_svm_decision_boundary(clf, sample_features, class_labels):
    """
    Visualizes the decision boundary of an SVM classifier for the two most important features.

    Args:
        clf (SVC): The trained SVM classifier.
        sample_features (DataFrame): The feature set used for training.
        class_labels (Series): The class labels corresponding to the feature set.
    """

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

def algorithm_metrics(clf, sample_features, true_labels):
    """
    Computes and displays accuracy, classification report, and confusion matrix for a classifier.

    Args:
        clf (Classifier): The trained classifier.
        sample_features (DataFrame): The feature set used for predictions.
        true_labels (Series): The true class labels for the feature set.
    """
    predict = clf.predict(sample_features)
    accuracy_dt = accuracy_score(predict, true_labels)
    print(f"- Accuracy : {accuracy_dt:.4f}")

    print("\n- Classification report:")
    print(classification_report(predict, true_labels, target_names=['M', 'R']))

    print("\n- Confusion Matrix:")
    cm_dt = confusion_matrix(predict, true_labels)
    cm_df_dt = pd.DataFrame(cm_dt, index=['Real M', 'Real R'], columns=['Predicted M', 'Predicted R'])
    print(cm_df_dt)

def SVM(x_train, x_test, y_train, y_test, class_names, kernel_func='linear'):
    """
    Trains and evaluates a Support Vector Machine (SVM) classifier.

    This function fits an SVM model on the training data, optionally visualizes 
    decision boundaries (if linear) or confusion matrices (if poly), and 
    prints performance metrics on the test set.
    """

    clf = svm.SVC(probability=True, kernel=kernel_func)
    clf.fit(x_train, y_train)

    if kernel_func == 'linear' and visualize_tree:
        visualize_svm_decision_boundary(clf, x_train, y_train)

    if kernel_func == 'poly':
        visualize_confusion_matrix(clf, x_test, y_test, class_names)
    
    if PRINT_REPORT:
        print(f'\nSVM with {kernel_func} kernel:')
        algorithm_metrics(clf, x_test, y_test, class_names)

    print(f'\nSVM with {kernel_func} kernel:')
    print_sample_row(clf, x_test)
    
    
def visualize_confusion_matrix(clf, x_test, y_test, class_names):
    """
    Generates and plots the confusion matrix for the provided classifier using test data.
    """
    predictions = clf.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - SVM (poly) for sonar testing set")
    plt.show()

def decision_tree(sample_features, class_labels, visualize_tree):
    """
    Implements decision tree classification and optionally visualizes the tree.

    Args:
        sample_features (DataFrame): The feature set used for training.
        class_labels (Series): The class labels corresponding to the feature set.
        visualize_tree (bool): Flag to visualize the decision tree.
    """
    clf = tree.DecisionTreeClassifier()
    clf.fit(sample_features, class_labels)
    
    visualize_decision_tree(clf) if visualize_tree else None
    if PRINT_REPORT:
        print('\nDecision Tree:')
        algorithm_metrics(clf, sample_features, class_labels)

    print('\nDecision Tree:')
    print_sample_row(clf, sample_features)

def print_sample_row(clf, sample_features):
    """
    Prints predictions and probabilities for a specific sample row.

    Args:
        clf (Classifier): The trained classifier.
        sample_features (DataFrame): The feature set containing the sample row.
    """
    single_row_from_features = sample_features.iloc[[SAMPLE_ROW_INDEX]]
    print(f'Prediction for sample {SAMPLE_ROW_INDEX+1}: {clf.predict(single_row_from_features)[0]}')
    print(f'Prediction probabilities for sample {SAMPLE_ROW_INDEX+1}: {clf.predict_proba(single_row_from_features)}')


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, class_names = read_sonar_data()

    print(f"Training Data: {x_train.shape}")
    print(f"Test Data: {x_test.shape}")
    print(f"Classes: {class_names}")

    # decision_tree(x_train, x_test, y_train, y_test, class_names, visualize_tree=VISUALIZE_TREES)
    SVM(x_train, x_test, y_train, y_test, class_names, kernel_func='poly')

