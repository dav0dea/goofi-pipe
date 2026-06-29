"""_node_directory must iterate a SNAPSHOT of the live node set.

The supervisor thread rebuilds the nd('name') directory (restart_node ->
_broadcast_node_directory -> _node_directory) while bridge add_node/remove_node
mutate self.nodes on an executor thread with no shared lock. Iterating the live
container then raises 'dictionary changed size during iteration' — the same
failure serialize_patch and _broadcast_node_directory already guard against with
list(self.nodes).
"""
import types

from goofi.manager import Manager


class _MutatingNodes:
    """A NodeContainer stand-in whose first __getitem__ inserts a key — simulating a
    concurrent add landing while _node_directory iterates."""

    def __init__(self, items):
        self._d = dict(items)
        self._tripped = False

    def __iter__(self):
        return iter(self._d.keys())  # live keys view, like NodeContainer.__iter__

    def __getitem__(self, k):
        if not self._tripped:
            self._tripped = True
            self._d["__ghost__"] = self._d[k]  # size change mid-iteration
        return self._d[k]


def test_node_directory_snapshots_nodes_for_safe_iteration():
    nodes = _MutatingNodes(
        {
            "u1": types.SimpleNamespace(name="osc0", node_id="id1"),
            "u2": types.SimpleNamespace(name="buf0", node_id="id2"),
        }
    )
    stub = types.SimpleNamespace(nodes=nodes)
    # Must not raise RuntimeError: dictionary changed size during iteration.
    directory = Manager._node_directory(stub)
    assert directory["osc0"] == "id1"
    assert directory["buf0"] == "id2"
