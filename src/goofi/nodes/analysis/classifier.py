from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class Classifier(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "probs": DataType.ARRAY,
            "feature_importances": DataType.ARRAY,  # for RandomForest and LogisticRegression
        }

    def config_params():
        return {
            "classification": {
                "n_states": IntParam(2, 1, 10),
                "current_state": IntParam(1, 1, 10),
                "add_to_training": BoolParam(
                    False, doc="Add the incoming data to the training set, must be in the shape (n_features, n_samples)"
                ),
                "train": BoolParam(False, trigger=True, doc="Train the classifier"),
                "classifier_choice": StringParam(
                    "SVM", options=["NaiveBayes", "SVM", "RandomForest", "LogisticRegression", "KNeighbors"]
                ),
                "clear_training": BoolParam(False, trigger=True, doc="Clear the training set"),
                "load_data": BoolParam(False, trigger=True),
                "save_data": BoolParam(False, trigger=True),
                "data_file": StringParam("goofi_clf_data", doc="File name for saving/loading training data"),
            },
            "NaiveBayes": {"var_smoothing": FloatParam(1e-9, 1e-12, 1e-6)},
            "SVM": {
                "C": FloatParam(1.0, 0.1, 10.0, doc="Penalty parameter C of the error term"),
                "kernel": StringParam(
                    "rbf",
                    options=["linear", "poly", "rbf", "sigmoid"],
                    doc="Specifies the kernel type to be used in the algorithm",
                ),
                "gamma": StringParam(
                    "scale", options=["scale", "auto"], doc="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’"
                ),
            },
            "RandomForest": {
                "n_estimators": IntParam(100, 10, 1000, doc="Number of trees in the forest"),
                "max_depth": IntParam(3, 1, 100, doc="Maximum depth of the tree"),
                "min_samples_split": IntParam(10, 2, 100, doc="Minimum number of samples required to split a node"),
            },
            "LogisticRegression": {"C": FloatParam(1.0, 0.1, 10.0, doc="Inverse of regularization strength")},
            "KNeighbors": {
                "n_neighbors": IntParam(5, 1, 20, doc="Number of neighbors to use by default for kneighbors queries")
            },
        }

    def setup(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        self.RandomForestClassifier = RandomForestClassifier
        self.LogisticRegression = LogisticRegression
        self.GaussianNB = GaussianNB
        self.KNeighborsClassifier = KNeighborsClassifier
        self.SVC = SVC

        self.training_data = []
        self.training_labels = []
        self.classifier = None
        self.classifier_trained = False

    def process(self, data: Data):
        if data is None:
            return None

        flattened = False
        if data.data.ndim > 1:
            flattened = True
        flat_data = data.data.flatten()

        self.data_file_path = join(self.assets_path, self.params.classification.data_file.value)

        if self.params.classification.add_to_training.value:
            if len(self.training_data) > 0 and len(flat_data) != len(self.training_data[0]):
                raise ValueError(
                    f"Incoming data size {len(flat_data)} does not match training data size {len(self.training_data[0])}."
                )
            self.training_data.append(flat_data)
            self.training_labels.append(self.params.classification.current_state.value)
            self.classifier = None
            self.classifier_trained = False
            print("Added to training set.")  # Debug statement
            return None

        if self.params.classification.load_data.value:
            try:
                loaded_data = np.load(self.data_file_path + (".npz" if not self.data_file_path.endswith(".npz") else ""))
                self.training_data = loaded_data["training_data"].tolist()
                self.training_labels = loaded_data["training_labels"].tolist()
                print(f"Loaded training data from {self.data_file_path}.")  # Debug statement
            except Exception as e:
                print(f"Error during loading data: {e}")  # Debug statement
                return None

            # Automatically train when data is loaded
            self.params.classification.train.value = True

        if self.params.classification.save_data.value:
            try:
                np.savez(self.data_file_path, training_data=self.training_data, training_labels=self.training_labels)
                print(f"Saved training data to {self.data_file_path}.")  # Debug statement
            except Exception as e:
                print(f"Error during saving data: {e}")  # Debug statement
                return None

        if self.params.classification.clear_training.value:
            self.training_data = []
            self.training_labels = []
            self.classifier = None
            self.classifier_trained = False

        if len(self.training_data) == 0:
            print("No training data.")
            return None

        if self.classifier is None:
            classifier_choice = self.params.classification.classifier_choice.value

            if classifier_choice == "NaiveBayes":
                self.classifier = self.GaussianNB(var_smoothing=self.params.NaiveBayes.var_smoothing.value)
            elif classifier_choice == "SVM":
                self.classifier = self.SVC(
                    C=self.params.SVM.C.value,
                    kernel=self.params.SVM.kernel.value,
                    gamma=self.params.SVM.gamma.value,
                    probability=True,
                )
            elif classifier_choice == "RandomForest":
                self.classifier = self.RandomForestClassifier(
                    n_estimators=self.params.RandomForest.n_estimators.value,
                    max_depth=self.params.RandomForest.max_depth.value,
                    min_samples_split=self.params.RandomForest.min_samples_split.value,
                )
            elif classifier_choice == "LogisticRegression":
                self.classifier = self.LogisticRegression(C=self.params.LogisticRegression.C.value)
            elif classifier_choice == "KNeighbors":
                self.classifier = self.KNeighborsClassifier(n_neighbors=self.params.KNeighbors.n_neighbors.value)

        if self.params.classification.train.value:
            # check if there are enough samples for each class
            for i in range(1, self.params.classification.n_states.value + 1):
                if self.training_labels.count(i) < 2:
                    print(f"Not enough samples for class {i} in training set.")
                    return None
            try:
                print(
                    f"Training data shape: {len(self.training_data)} samples, {len(self.training_data[0]) if self.training_data else 0} features per sample."
                )  # Debug statement
                self.classifier.fit(self.training_data, self.training_labels)
                self.classifier_trained = True
            except Exception as e:
                print(f"Error during fitting: {e}")  # Debug statement

        # check if the classifier has been trained
        if self.classifier_trained is False:
            print("Classifier not trained.")
            return None

        probs = self.classifier.predict_proba(flat_data[None])

        # After the classifier has been trained
        feature_importances = self.get_feature_importances()

        # create metadata including the classifier, the size of the training set for each class, and the number of features
        meta = {"classifier": self.params.classification.classifier_choice.value}
        for i in range(1, self.params.classification.n_states.value + 1):
            meta[f"training_set_size_{i}"] = self.training_labels.count(i)
        meta["n_features"] = len(self.training_data[0]) if self.training_data else 0

        if feature_importances is None:
            return {"probs": (probs, meta), "feature_importances": None}

        meta_features = {}
        if "channels" in data.meta and not flattened:
            meta_features["channels"] = data.meta["channels"]
        if "sfreq" in data.meta:
            meta["sfreq"] = data.meta["sfreq"]
            meta_features["sfreq"] = data.meta["sfreq"]
        return {"probs": (probs, meta), "feature_importances": (np.array(feature_importances), meta_features)}

    def get_feature_importances(self):
        """Retrieve feature importances from the classifier."""
        if isinstance(self.classifier, self.RandomForestClassifier):
            return self.classifier.feature_importances_
        elif isinstance(self.classifier, self.LogisticRegression):
            return self.classifier.coef_[0]
        elif isinstance(self.classifier, self.SVC) and self.params.SVM.kernel.value == "linear":
            return self.classifier.coef_[0]
        else:
            print("Feature importances not available for the current classifier or configuration.")
            return None
