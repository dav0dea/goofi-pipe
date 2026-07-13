import time

import pytest

from goofi.registry import NodeSpec, build_catalog

from .test_registry_parity import load_or_skip

CATALOG, _ = build_catalog()

SKIP_NODES = ["AudioStream", "AudioOut", "Oscillator", "MidiCCout", "MidiCout"]

# The create/terminate smoke tests only check a node's lifecycle, not its
# processing. Spawn them with autotrigger off so no node free-runs its
# (side-effectful) process() during the test — e.g. VideoStream would otherwise
# grab the webcam on the first tick. Combined with VideoStream opening the camera
# lazily in process() (not setup()), this keeps the camera off during tests.
IDLE_PARAMS = {"common": {"autotrigger": False}}


@pytest.mark.parametrize("spec", CATALOG.values(), ids=lambda s: s.type)
def test_implement_init(spec: NodeSpec):
    node = load_or_skip(spec)
    # make sure the node uses the base class' __init__ and does not override it
    assert (
        node.__init__.__qualname__ == "Node.__init__"
    ), f"{node.__name__} should not override __init__. Use the setup() function to initialize the node instead."


@pytest.mark.parametrize("spec", CATALOG.values(), ids=lambda s: s.type)
def test_create_local(spec: NodeSpec, timeout: float = 20.0):
    if spec.type in SKIP_NODES:
        pytest.skip("Github Actions does not support audio devices.")
    node = load_or_skip(spec)

    # TODO: a 20 second timeout is too much, is there a way to kill nodes faster?
    ref, n = node.create_local(initial_params=IDLE_PARAMS)
    ref.terminate()

    start = time.time()
    while time.time() - start < timeout and n.alive:
        time.sleep(0.01)
    assert not n.alive, f"Node {node.__name__} needed more than {timeout}s to terminate."


@pytest.mark.parametrize(
    "spec", [s for s in CATALOG.values() if not s.no_multiprocessing], ids=lambda s: s.type
)
def test_create(spec: NodeSpec):
    if spec.type in SKIP_NODES:
        pytest.skip("Github Actions does not support audio devices.")
    node = load_or_skip(spec)

    node.create(initial_params=IDLE_PARAMS).terminate()


@pytest.mark.parametrize("spec", CATALOG.values(), ids=lambda s: s.type)
def test_category(spec: NodeSpec):
    node = load_or_skip(spec)
    cat = node.category()
    assert cat == spec.category, f"AST category and class category diverge for {spec.type}."
    assert isinstance(cat, str), f"Category of {node.__name__} should be a string."
    assert cat != "nodes", (
        f"Category of {node.__name__} should not be 'nodes'. The node should be placed in "
        f"'goofi/nodes/<category>/'."
    )
    assert cat != "goofi", (
        f"Category of {node.__name__} should not be 'goofi'. The node should be placed in "
        f"'goofi/nodes/<category>/'."
    )
