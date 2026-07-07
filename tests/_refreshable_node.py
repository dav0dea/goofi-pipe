"""A node with a refreshable StringParam, for the refresh_param end-to-end test.
Its refresh method returns a fixed 'live' list distinct from the seed options, so
the test can watch the ref's options change from the node's push."""
from goofi.data import DataType
from goofi.node import Node
from goofi.params import StringParam


class Refreshable(Node):
    def config_params():
        return {"g": {"pick": StringParam("seed", options=["seed"], refresh="_reload")}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def _reload(self):
        return ["live-a", "live-b"]

    def process(self):
        return {"out": (0, {})}
