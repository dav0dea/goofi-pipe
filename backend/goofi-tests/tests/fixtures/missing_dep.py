import definitely_not_installed_pkg  # noqa: F401  -> probe import fails -> grey-out

import goofi


class Broken(goofi.Node):
    manifest = goofi.Manifest(inputs={"data": goofi.DataType.ARRAY})
