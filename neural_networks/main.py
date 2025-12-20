"""
This script implements TensoFlow neural networks for clustering Cifar10, FashionMnist and sonar readings datasets. 
It includes functionalities for data visualization, algorithm metrics evaluation, and prediction analysis.

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
Run the script from the command line in the same directory as the main.py file. 

Dataset sources:
- Ecoli: https://github.com/MachineLearningBCAM/Datasets/tree/main
- fashion MNIST: https://www.tensorflow.org/tutorials/keras/classification
- CIFAR10: https://www.tensorflow.org/datasets/catalog/cifar10
- sonar readings: https://machinelearningmastery.com/standard-machine-learning-datasets/ 

This project is built using TensorFlow (https://www.tensorflow.org/) licensed under the Apache License, Version 2.0.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns


class Classifier:
    """
    A base class for building, training, and evaluating TensorFlow neural network models.

    This class serves as a template for specific dataset classifiers. It handles 
    common boilerplate operations such as checking model readiness, training loops, 
    evaluation, prediction, and visualization of confusion matrices. 

    Subclasses are expected to implement `setup_model` and `load_data`.
    """
    def __init__(self):
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.labels = []

    def setup_model(self):
        """Instantiate and compile the underlying model."""
        raise NotImplementedError("Subclasses must implement setup_model().")

    def load_data(self):
        """Load and preprocess the dataset."""
        raise NotImplementedError("Subclasses must implement load_data().")

    def _ensure_model_ready(self):
        if self.model is None:
            self.setup_model()
        if self.x_train is None or self.x_test is None:
            self.load_data()
        if self.model is None or self.x_train is None or self.x_test is None:
            raise ValueError("Model or data not properly initialized.")

    def train(self, epochs=5, validate_data=False, **fit_kwargs):
        """Train the classifier on the prepared dataset."""
        self._ensure_model_ready()
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data not initialized.")
        return self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test) if validate_data else None,
            **fit_kwargs,
        )

    def predict(self, index, show_plot=True):
        """Produce predictions for given input data."""
        self._ensure_model_ready()
        if self.x_test is None or self.y_test is None:
            raise ValueError("Prediction data not initialized.")
        image = self.x_test[index]
        true_label = self.y_test[index]
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        guessed_class = int(np.argmax(prediction))
        if show_plot:
            plt.figure(figsize=(6, 3))
            if image.ndim == 3 and image.shape[2] == 3:
                plt.imshow(image)  # RGB
            else:
                plt.imshow(image, cmap=plt.cm.binary)  # Grayscale
            plt.title(
                f"True: {self.class_names[true_label]}\nNetwork thinks: {self.class_names[guessed_class]}"
            )
            plt.xlabel(f"Confidence: {100 * np.max(prediction):.2f}%")
            plt.colorbar()
            plt.show()
        return guessed_class, prediction

    def show_confusion_matrix(self, title="Confusion Matrix"):
        """
        Generates and plots the confusion matrix for the provided classifier using test data.
        """
        self._ensure_model_ready()

        predictions = self.model.predict(self.x_test, verbose=0)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.y_test

        cm = tf.math.confusion_matrix(
            labels=true_classes, predictions=predicted_classes
        ).numpy()

        plt.figure(figsize=(8, 8))

        sns.heatmap(cm, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names, 
                yticklabels=self.class_names)

        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plt.show()


class FashionMnist(Classifier):
    """
    A classifier for the Fashion MNIST dataset using a simple Feed-Forward Neural Network.

    This class loads the standard Fashion MNIST dataset (grayscale images of clothing),
    normalizes the pixel values to the 0-1 range, and trains a basic Multi-Layer 
    Perceptron (MLP) with one hidden layer to classify the 10 categories.
    """
    def __init__(self):
        super().__init__()
        self.class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def load_data(self):
        """Load and preprocess the Fashion MNIST dataset."""
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            fashion_mnist.load_data()
        )
        self.x_train, self.x_test = (
            self.x_train / 255.0,
            self.x_test / 255.0,
        )

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


class Cifar10(Classifier):
    """
    A baseline classifier for the CIFAR-10 dataset using a Feed-Forward Neural Network.

    This class flattens the 32x32x3 RGB images into 1D vectors and processes them through
    a dense network.
    """
    def __init__(self):
        super().__init__()
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def load_data(self):
        cifar10 = tf.keras.datasets.cifar10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # In CIFAR, labels are in a 2D array [[6], [9]...].
        # We need to flatten them to [6, 9...] for convenience.
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    64, activation="relu" #only 64 neurons for this layer
                ), 
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


class DeepCifar10(Cifar10):
    """
    An advanced classifier for the CIFAR-10 dataset using a Convolutional Neural Network (CNN).

    Inherits from the `Cifar10` class but overrides `setup_model`. The Convolutional Neural Network (CNN) 
    includes multiple Conv2D and MaxPooling2D layers, followed by Dropout for 
    regularization, offering significantly higher accuracy than the base MLP model.
    """
    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),  #overfitting protection
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


class Sonar(Classifier):
    """
    A binary classifier for the Sonar Mines vs. Rocks dataset.

    This class processes sonar signal data loaded from a CSV file. It handles 
    parsing the string labels ('M' for Mine, 'R' for Rock) into integer format,
    performs an 80/20 train-test split, and trains a deep dense network to 
    distinguish between the two classes.
    """
    def __init__(self):
        super().__init__()

    def load_data(self):
        df = pd.read_csv("data/sonar.csv", header=None)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        y_column = df.iloc[:, -1]

        unique_classes = sorted(y_column.unique())
        self.class_names = unique_classes

        label_map = {name: i for i, name in enumerate(unique_classes)}

        df.iloc[:, -1] = df.iloc[:, -1].map(label_map)

        all_x = df.iloc[:, :-1].values.astype("float32")
        all_y = df.iloc[:, -1].values.astype("int32")

        num_samples = len(df)
        train_size = int(0.8 * num_samples)

        self.x_train = all_x[:train_size]
        self.y_train = all_y[:train_size]

        self.x_test = all_x[train_size:]
        self.y_test = all_y[train_size:]

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(60,)),
                tf.keras.layers.Dense(60, activation="relu"),
                tf.keras.layers.Dense(30, activation="relu"),
                tf.keras.layers.Dense(15, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def predict(self, index, show_plot=True):
        self._ensure_model_ready()

        sample = self.x_test[index]
        true_label = self.y_test[index]

        prediction = self.model.predict(np.expand_dims(sample, axis=0), verbose=0)
        guessed_class = int(np.argmax(prediction))

        if show_plot:
            plt.figure(figsize=(10, 2))
            plt.imshow([sample], aspect="auto", cmap="viridis")

            is_correct = true_label == guessed_class
            color = "green" if is_correct else "red"

            plt.title(
                f"True: {self.class_names[true_label]} | Network thinks: {self.class_names[guessed_class]}",
                color=color,
                fontweight="bold",
            )
            plt.xlabel(f"Confidence: {np.max(prediction) * 100:.1f}%")
            plt.yticks([])
            plt.colorbar(label="Signal")
            plt.show()

        return guessed_class, prediction


class Ecoli(Classifier):
    """
    A multi-class classifier for the Ecoli Protein Localization dataset.

    This class handles the loading and preprocessing of biological data from a CSV.
    It extracts 7 specific features (mcg, gvh, lip, etc.), maps text labels to 
    integers, and trains a neural network to predict the cellular localization 
    site of proteins.
    """
    def __init__(self):
        super().__init__()

    def load_data(self):
        df = pd.read_csv("data/ecoli.csv", header=2)  # Skip first two header rows
        self.feature_names = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]

        # Ignore first column (Sequence Name)
        df = df.iloc[:, 1:]

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        y_column = df.iloc[:, -1]
        self.class_names = sorted(y_column.unique())

        label_map = {name: i for i, name in enumerate(self.class_names)}

        df.iloc[:, -1] = df.iloc[:, -1].map(label_map)

        all_x = df.iloc[:, :-1].values.astype("float32")
        all_y = df.iloc[:, -1].values.astype("int32")

        self.num_features = all_x.shape[1]
        self.num_classes = len(self.class_names)

        train_size = int(0.8 * len(df))

        self.x_train = all_x[:train_size]
        self.y_train = all_y[:train_size]

        self.x_test = all_x[train_size:]
        self.y_test = all_y[train_size:]

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(self.num_features,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def predict(self, index, show_plot=True):
        self._ensure_model_ready()

        sample = self.x_test[index]
        true_label = self.y_test[index]

        prediction = self.model.predict(np.expand_dims(sample, axis=0), verbose=0)
        guessed_class = int(np.argmax(prediction))

        if show_plot:
            plt.figure(figsize=(8, 4))
            plt.bar(range(self.num_features), sample, color="teal")

            is_correct = true_label == guessed_class
            color = "green" if is_correct else "red"

            true_name = self.class_names[true_label]
            guessed_name = self.class_names[guessed_class]

            plt.title(
                f"True: {true_name} | Network thinks: {guessed_name}",
                color=color,
                fontweight="bold",
            )
            plt.xlabel("Feature indices (0-6)")
            plt.ylabel("Value")
            plt.grid(axis="y", alpha=0.3)
            plt.xticks(range(self.num_features), self.feature_names)

            confidence = np.max(prediction) * 100
            plt.suptitle(f"Network confidence: {confidence:.2f}%", y=0.95)

            plt.show()

        return guessed_class, prediction


def setup_classifier(c, epochs=5):
    """
    Define layers on the network, load dataset and train it for a number of times.
    """
    c.setup_model()
    c.load_data()
    c.train(epochs=epochs)
    return c


def compare_classifiers(c1: Classifier, c2: Classifier, c1_epochs=5, c2_epochs=5):
    """
    Choose two classifiers and compare them using different number of training iterations
    """
    histories = {}

    c1_time = tf.timestamp()
    c1.load_data()
    c1.setup_model()
    hist_c1 = c1.train(c1_epochs, validate_data=True, verbose=1).history
    c1_time = tf.timestamp() - c1_time
    histories["c1"] = hist_c1

    c2_time = tf.timestamp()
    c2.load_data()
    c2.setup_model()
    hist_c2 = c2.train(c2_epochs, validate_data=True, verbose=1).history
    c2_time = tf.timestamp() - c2_time
    histories["c2"] = hist_c2

    plt.figure(figsize=(10, 5))

    plt.plot(
        histories["c1"]["val_accuracy"],
        linestyle="--",
        label="Model" + type(c1).__name__,
        color="red",
    )
    plt.plot(
        histories["c2"]["val_accuracy"],
        linestyle="-",
        label="Model" + type(c2).__name__,
        color="blue"
    )

    plt.title(
        f"Accuracy comparison: Model {type(c1).__name__} (time: {c1_time:.2f}s) vs Model {type(c2).__name__} (time: {c2_time:.2f}s)"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # compare_classifiers(Cifar10(), DeepCifar10(), c1_epochs=25, c2_epochs=5)

    sonar = setup_classifier(Sonar())
    sonar.show_confusion_matrix(title="Confusion matrix - NN for sonar testing set")

    fashion = setup_classifier(FashionMnist())
    fashion.show_confusion_matrix(title="fashion")
