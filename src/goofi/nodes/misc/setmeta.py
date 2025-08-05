from goofi.data import Data, DataType
from goofi.node import Node


class SetMeta(Node):
    """
    This node sets a metadata key-value pair on an input array. The value is type-cast based on a specified type before assignment. The data remains unchanged.

    Inputs:
    - array: Input array data with accompanying metadata.

    Outputs:
    - out: The same array data with the updated metadata.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"meta": {"key": "key", "value": "value", "type": "float"}}

    def process(self, array: Data):
        if array is None:
            return None

        val = self.params.meta.value.value

        if self.params.meta.type.value == "int":
            val = int(val)
        elif self.params.meta.type.value == "float":
            val = float(val)
        elif self.params.meta.type.value == "bool":
            val = bool(val)
        elif self.params.meta.type.value == "str":
            val = str(val)
        else:
            raise ValueError(f"Unknown type {self.params.meta.type.value}")

        array.meta[self.params.meta.key.value] = val
        return {"out": (array.data, array.meta)}
