import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class Classifier:
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

    def train(self, epochs=5, **fit_kwargs):
        """Train the classifier on the prepared dataset."""
        self._ensure_model_ready()
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data not initialized.")
        return self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            **fit_kwargs,
        )

    def evaluate(self, **eval_kwargs):
        """Run evaluation against the holdout dataset."""
        self._ensure_model_ready()
        if self.x_test is None or self.y_test is None:
            raise ValueError("Evaluation data not initialized.")
        return self.model.evaluate(
            self.x_test,
            self.y_test,
            **eval_kwargs,
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


class FashionMnist(Classifier):
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
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
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

        # IMPORTANT: In CIFAR, labels are in a 2D array [[6], [9]...].
        # We need to flatten them to [6, 9...] for convenience.
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),  # scaling down 2x
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),  # scaling down 2x
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.Flatten(),  # Flatten to 1D
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )


class Sonar(Classifier):
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
                tf.keras.layers.InputLayer(input_shape=(60,)),
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
    def __init__(self):
        super().__init__()

    def load_data(self):
        df = pd.read_csv("data/ecoli.csv", header=2)  # Skip first two header rows
        self.feature_names = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]

        # Ignore first column (Sequence Name)
        df = df.iloc[:, 1:]

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        y_column = df.iloc[:, -1]
        unique_classes = sorted(y_column.unique())
        self.class_names = unique_classes

        label_map = {name: i for i, name in enumerate(unique_classes)}

        df.iloc[:, -1] = df.iloc[:, -1].map(label_map)

        all_x = df.iloc[:, :-1].values.astype("float32")
        all_y = df.iloc[:, -1].values.astype("int32")

        self.num_features = all_x.shape[1]
        self.num_classes = len(unique_classes)

        train_size = int(0.8 * len(df))

        self.x_train = all_x[:train_size]
        self.y_train = all_y[:train_size]

        self.x_test = all_x[train_size:]
        self.y_test = all_y[train_size:]

    def setup_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.num_features,)),
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


if __name__ == "__main__":
    classifier = Cifar10()
    classifier.setup_model()
    classifier.train(epochs=5)
    classifier.evaluate()
    classifier.predict(index=7)
