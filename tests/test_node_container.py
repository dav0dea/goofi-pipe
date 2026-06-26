import pytest

from goofi.manager import NodeContainer

from .utils import DummyNode


def test_creation():
    cont = NodeContainer()
    assert len(cont) == 0


def test_assignment():
    cont = NodeContainer()
    # the container shouldn't allow item assignment
    with pytest.raises(TypeError):
        cont["test"] = 1


def test_contains():
    # The container is keyed by stable uid (display-name generation lives in
    # Manager.add_node, not here).
    cont = NodeContainer()
    assert "u1" not in cont, "Empty container shouldn't contain anything"

    ref = DummyNode.create_local()[0]
    cont.add_node("u1", ref)
    assert "u1" in cont, "Added node but container doesn't contain it"
    ref.terminate()


def test_add_node():
    cont = NodeContainer()

    # adding a node should increase the length of the container; the uid is the key
    ref1 = DummyNode.create_local()[0]
    assert cont.add_node("u1", ref1) == "u1"
    assert len(cont) == 1, "Added node but length didn't increase"
    assert "u1" in cont

    # a distinct uid is a distinct entry
    ref2 = DummyNode.create_local()[0]
    cont.add_node("u2", ref2)
    assert len(cont) == 2
    assert "u2" in cont

    # re-adding the SAME uid is an error (no silent re-key — Manager mints unique uids)
    with pytest.raises(KeyError):
        cont.add_node("u1", DummyNode.create_local()[0])

    # check type validation
    ref3 = DummyNode.create_local()[0]
    with pytest.raises(ValueError):
        cont.add_node(1, ref3)
    with pytest.raises(ValueError):
        cont.add_node("u9", None)

    ref1.terminate()
    ref2.terminate()
    ref3.terminate()


def test_remove_node():
    cont = NodeContainer()
    cont.add_node("u1", DummyNode.create_local()[0])

    # removing a node should decrease the length of the container
    cont.remove_node("u1")
    assert len(cont) == 0, "Removed node but length didn't decrease"

    # check failure cases
    with pytest.raises(KeyError):
        cont.remove_node("u1")
    with pytest.raises(KeyError):
        cont.remove_node(None)
    with pytest.raises(KeyError):
        cont.remove_node(1)
