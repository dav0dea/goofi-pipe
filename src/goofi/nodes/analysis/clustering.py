import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Clustering(Node):
    """
    This node performs clustering on input data using either the KMeans or Agglomerative Clustering algorithms. It assigns each data point in the input matrix to a cluster, and for KMeans, also computes the centers of each cluster.

    Inputs:
    - matrix: An array-like structure containing the data to be clustered. Each row should represent a sample, and each column a feature.

    Outputs:
    - cluster_labels: An array with a cluster label for each input sample, indicating its assigned cluster.
    - cluster_centers: An array of the centers of the clusters (only produced when using the KMeans algorithm).
    """

    def config_input_slots():
        return {"matrix": DataType.ARRAY}

    def config_output_slots():
        return {
            "cluster_labels": DataType.ARRAY,
            "cluster_centers": DataType.ARRAY,  # for KMeans
        }

    def config_params():
        return {
            "Clustering": {
                "algorithm": StringParam("KMeans", options=["KMeans", "Agglomerative"]),
                "n_clusters": IntParam(2, 1, 20),  # Assuming max clusters as 20 for simplicity
                "max_iter": IntParam(300, 1, 1000, doc="Maximum number of iterations for KMeans"),
                "tolerance": FloatParam(0.0001, 0.0001, 0.1, doc="Tolerance for KMeans convergence"),
                "affinity": StringParam("euclidean", options=["euclidean", "l1", "l2", "manhattan", "cosine"]),
                "linkage": StringParam("ward", options=["ward", "complete", "average", "single"]),
            }
        }

    def setup(self):
        from sklearn.cluster import AgglomerativeClustering, KMeans

        self.AgglomerativeClustering = AgglomerativeClustering
        self.KMeans = KMeans

    def process(self, matrix: Data):
        if matrix is None:
            return None
        max_iter = self.params.Clustering["max_iter"].value
        tolerance = self.params.Clustering["tolerance"].value
        if self.params.Clustering["algorithm"].value == "KMeans":
            model = self.KMeans(n_clusters=self.params.Clustering["n_clusters"].value, max_iter=max_iter, tol=tolerance)
            labels = model.fit_predict(matrix.data)
            centers = model.cluster_centers_
            labels = labels.astype(np.float64)
            return {
                "cluster_labels": (labels, matrix.meta),
                "cluster_centers": (centers, matrix.meta),
            }

        elif self.params.Clustering["algorithm"].value == "Agglomerative":
            model = self.AgglomerativeClustering(
                n_clusters=self.params.Clustering["n_clusters"].value,
                affinity=self.params.Clustering["affinity"].value,
                linkage=self.params.Clustering["linkage"].value,
            )
            labels = model.fit_predict(matrix.data)
            # put the labels as float
            labels = labels.astype(np.float64)
            return {"cluster_labels": (labels, matrix.meta)}
