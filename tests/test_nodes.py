import time
from typing import Type

import pytest

from goofi.node import Node
from goofi.node_helpers import list_nodes

SKIP_NODES = ["AudioStream", "AudioOut", "MidiCCout", "MidiCout"]

# The create/terminate smoke tests only check a node's lifecycle, not its
# processing. Spawn them with autotrigger off so no node free-runs its
# (side-effectful) process() during the test — e.g. VideoStream would otherwise
# grab the webcam on the first tick. Combined with VideoStream opening the camera
# lazily in process() (not setup()), this keeps the camera off during tests.
IDLE_PARAMS = {"common": {"autotrigger": False}}


@pytest.mark.parametrize("node", list_nodes())
def test_implement_init(node: Type[Node]):
    # make sure the node uses the base class' __init__ and does not override it
    assert (
        node.__init__.__qualname__ == "Node.__init__"
    ), f"{node.__name__} should not override __init__. Use the setup() function to initialize the node instead."


@pytest.mark.parametrize("node", list_nodes())
def test_create_local(node: Type[Node], timeout: float = 20.0):
    if node.__name__ in SKIP_NODES:
        pytest.skip("Github Actions does not support audio devices.")

    # TODO: a 20 second timeout is too much, is there a way to kill nodes faster?
    ref, n = node.create_local(initial_params=IDLE_PARAMS)
    ref.terminate()

    start = time.time()
    while time.time() - start < timeout and n.alive:
        time.sleep(0.01)
    assert not n.alive, f"Node {node.__name__} needed more than {timeout}s to terminate."


@pytest.mark.parametrize("node", [n for n in list_nodes() if not n.NO_MULTIPROCESSING])
def test_create(node: Type[Node]):
    if node.__name__ in SKIP_NODES:
        pytest.skip("Github Actions does not support audio devices.")

    node.create(initial_params=IDLE_PARAMS).terminate()


@pytest.mark.parametrize("node", list_nodes())
def test_category(node: Type[Node]):
    cat = node.category()
    assert isinstance(cat, str), f"Category of {node.__name__} should be a string."
    assert cat != "nodes", (
        f"Category of {node.__name__} should not be 'nodes'. The node should be placed in "
        f"'goofi/nodes/<category>/{node.__name__.lower()}.py'."
    )
    assert cat != "goofi", (
        f"Category of {node.__name__} should not be 'goofi'. The node should be placed in "
        f"'goofi/nodes/<category>/{node.__name__.lower()}.py'."
    )


@pytest.mark.parametrize("node", list_nodes())
def test_node_filename(node: Type[Node]):
    fname = node.__module__.split(".")[-1]
    # TODO: we might want to loosen this restriction in the future
    assert (
        fname == node.__name__.lower()
    ), f"Filename of node '{node.__name__}' should be '{node.__name__.lower()}.py' (TODO: maybe loosen this restriction)."
