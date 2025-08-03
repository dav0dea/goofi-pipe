import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DimensionalityReduction(Node):
    """
    Performs dimensionality reduction on array data using one of several algorithms (PCA, t-SNE, or UMAP), reducing high-dimensional input data to a lower-dimensional representation. This node can also optionally transform new incoming data samples into the previously computed low-dimensional space, when supported by the selected algorithm.

    Inputs:
    - data: The original array data to be reduced in dimensionality. Must be 2D.
    - new_data: New array data samples to be projected into the computed lower-dimensional space using the fitted model.

    Outputs:
    - transformed: The array data transformed into the lower-dimensional space, along with updated metadata.
    - new_components: The new data samples transformed into the same lower-dimensional space, with updated metadata. Only provided if new_data is given and supported by the selected method.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY, "new_data": DataType.ARRAY}

    def config_output_slots():
        return {
            "transformed": DataType.ARRAY,
            "new_components": DataType.ARRAY,
        }

    def config_params():
        return {
            "dim_red": {
                "reset": BoolParam(False, trigger=True, doc="Reset the buffer"),
                "method": StringParam("PCA", options=["PCA", "t-SNE", "UMAP"], doc="Dimensionality reduction method"),
                "n_components": IntParam(2, 1, 10, doc="Number of output dimensions"),
            },
            "umap": {
                "num_neighbors": IntParam(15, 2, 100, doc="Number of UMAP neighbors"),
                "metric": StringParam("euclidean", options=UMAP_METRICS, doc="Distance metric for UMAP"),
            },
            "tsne": {"perplexity": FloatParam(30.0, 5.0, 50.0, doc="t-SNE perplexity")},
        }

    def setup(self):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from umap import UMAP

        self.tsne_cls = TSNE
        self.pca_cls = PCA
        self.umap_cls = UMAP

        self.model = None
        self.components = None
        self.meta = None

    def process(self, data: Data, new_data: Data):
        if data is None:
            return None

        method = self.params.dim_red.method.value
        data_array = np.squeeze(data.data)

        if self.params.dim_red.reset.value:
            self.model = None
            self.components = None
            self.meta = None

        if self.components is not None:
            new_components = None
            if new_data is not None and self.model is not None:
                if method == "t-SNE":
                    raise ValueError("The t-SNE algorithm does not support transforming new data")

                new_data_arr = new_data.data
                if new_data_arr.ndim == 1:
                    new_data_arr = new_data_arr.reshape(1, -1)
                new_components = self.model.transform(new_data_arr).squeeze()

                new_meta = new_data.meta
                if "channels" in new_meta and "dim1" in new_meta["channels"]:
                    del new_meta["channels"]["dim1"]

            return {
                "transformed": (self.components, self.meta),
                "new_components": (new_components, new_meta) if new_components is not None else None,
            }

        if data_array.ndim != 2:
            raise ValueError("Data must be 2D")

        n_components = int(self.params.dim_red.n_components.value)

        self.meta = data.meta
        if "channels" in self.meta and "dim1" in self.meta["channels"]:
            del self.meta["channels"]["dim1"]

        new_components = None
        if method == "PCA":
            self.model = self.pca_cls(n_components=n_components)
            self.components = self.model.fit_transform(data_array)

            if new_data is not None:
                new_components = self.model.transform(new_data.data)
                new_meta = new_data.meta
                if "channels" in new_meta and "dim1" in new_meta["channels"]:
                    del new_meta["channels"]["dim1"]

        elif method == "t-SNE":
            self.model = self.tsne_cls(
                n_components=n_components,
                perplexity=self.params.tsne.perplexity.value,
                init="pca",
                random_state=42,
            )
            self.components = self.model.fit_transform(data_array)

        elif method == "UMAP":
            self.model = self.umap_cls(
                n_components=n_components,
                n_neighbors=self.params.umap.num_neighbors.value,
                metric=self.params.umap.metric.value,
                random_state=42,
            )
            self.components = self.model.fit_transform(data_array)

            if new_data is not None:
                new_data_arr = new_data.data
                if new_data_arr.ndim == 1:
                    new_data_arr = new_data_arr.reshape(1, -1)
                new_components = self.model.transform(new_data_arr).squeeze()

                new_meta = new_data.meta
                if "channels" in new_meta and "dim1" in new_meta["channels"]:
                    del new_meta["channels"]["dim1"]

        return {
            "transformed": (self.components, self.meta),
            "new_components": (new_components, self.meta) if new_components is not None else None,
        }


UMAP_METRICS = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "canberra",
    "braycurtis",
    "mahalanobis",
    "wminkowski",
    "seuclidean",
    "cosine",
    "correlation",
    "haversine",
    "hamming",
    "jaccard",
    "dice",
    "russelrao",
    "kulsinski",
    "ll_dirichlet",
    "hellinger",
    "rogerstanimoto",
    "sokalmichener",
    "sokalsneath",
    "yule",
]
