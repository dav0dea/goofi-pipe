import sys

import goofi

# The pygame banner, in miniature: a real dependency that greets stdout on import.
# The probe shares this child's stdout with the module, so anything written here
# prepends itself to the introspection payload unless the probe reroutes fd 1.
print("chatty 1.2.3 (SDL 2.28.4, Python 3.14.0)")
print("Hello from the community. https://example.invalid/contribute.html")
sys.stdout.write("and a bare write, with no newline")


class Chatty(goofi.Node):
    """A node whose dependency prints on import."""

    manifest = goofi.Manifest(
        inputs={"data": goofi.DataType.ARRAY},
        outputs={"out": goofi.DataType.ARRAY},
    )
