import definitely_not_installed_pkg  # noqa: F401  -> probe import fails -> grey-out

import goofi


class Broken(goofi.Node):
    def config_input_slots(self):
        return {"data": goofi.DataType.ARRAY}
