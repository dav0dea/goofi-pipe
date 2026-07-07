"""A node whose bootstrap raises an exception larger than the boot pipe's
capacity — used to prove the traceback is truncated before the one-shot send,
so the dying child never blocks in write() (which would wedge the node in
'creating' with no error surfaced)."""
from goofi.node import Node


class HugeError(Node):
    def config_params():
        raise RuntimeError("X" * 200_000)  # > the ~64 KiB pipe capacity

    def process(self):
        pass
