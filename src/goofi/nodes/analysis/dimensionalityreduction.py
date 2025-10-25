import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DimensionalityReduction(Node):

    def config_input_slots():
        return {"data": DataType.ARRAY, "new_data": DataType.ARRAY, "reset": DataType.ARRAY}

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
                "random_seed": IntParam(1234, 0, 10000, doc="Random seed for UMAP"),
            },
            "tsne": {
                "perplexity": FloatParam(30.0, 5.0, 50.0, doc="t-SNE perplexity"),
                "random_seed": IntParam(1234, 0, 10000, doc="Random seed for t-SNE"),
            },
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

    def process(self, data: Data, new_data: Data, reset: Data):
        if data is None:
            return None

        if (reset is not None and np.any(reset.data > 0)) or self.params.dim_red.reset.value:
            self.input_slots["reset"].clear()

            self.model = None
            self.components = None
            self.meta = None

        method = self.params.dim_red.method.value
        data_array = np.squeeze(data.data)

        if self.components is not None:
            new_components = None
            if new_data is not None and self.model is not None:
                if method == "t-SNE":
                    raise ValueError("The t-SNE algorithm does not support transforming new data")

                new_data_arr = new_data.data
                added_dim = False
                if new_data_arr.ndim == 1:
                    added_dim = True
                    new_data_arr = new_data_arr.reshape(1, -1)
                new_components = self.model.transform(new_data_arr)
                if added_dim:
                    new_components = new_components[0]

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
                random_state=self.params.tsne.random_seed.value,
            )
            self.components = self.model.fit_transform(data_array)

        elif method == "UMAP":
            self.model = self.umap_cls(
                n_components=n_components,
                n_neighbors=self.params.umap.num_neighbors.value,
                metric=self.params.umap.metric.value,
                random_state=self.params.umap.random_seed.value,
            )
            self.components = self.model.fit_transform(data_array)

            if new_data is not None:
                new_data_arr = new_data.data
                added_dim = False
                if new_data_arr.ndim == 1:
                    added_dim = True
                    new_data_arr = new_data_arr.reshape(1, -1)
                new_components = self.model.transform(new_data_arr)
                if added_dim:
                    new_components = new_components[0]

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
