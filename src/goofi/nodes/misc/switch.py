from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Switch(Node):
    """
    Selects and forwards one of several input arrays or strings to the output, based on the selector input and mode. The node routes either an array or a string input to the corresponding output.

    Inputs:
    - selector: Array input used to choose which input will be forwarded. The first element indicates selection (1, 2, or 3).
    - array1: First array input.
    - array2: Second array input.
    - array3: Third array input.
    - string1: First string input.
    - string2: Second string input.
    - string3: Third string input.

    Outputs:
    - array_out: Forwards the selected array input when the node is set to array mode. Only one array input is passed to the output depending on the selector value.
    - string_out: Forwards the selected string input when the node is set to string mode. Only one string input is passed to the output depending on the selector value.
    """

    def config_input_slots():
        return {
            "selector": DataType.ARRAY,  # Input 1: Selector array
            "array1": DataType.ARRAY,  # Input 2: First array input
            "array2": DataType.ARRAY,  # Input 3: Second array input
            "array3": DataType.ARRAY,  # Input 4: Third array input
            "string1": DataType.STRING,  # Input 5: First string input
            "string2": DataType.STRING,  # Input 6: Second string input
            "string3": DataType.STRING,  # Input 7: Third string input
        }

    def config_output_slots():
        return {
            "array_out": DataType.ARRAY,  # Output 1: Array output
            "string_out": DataType.STRING,  # Output 2: String output
        }

    def config_params():
        return {
            "settings": {
                "mode": StringParam("array", ["array", "string"]),  # Dropdown to choose mode
            }
        }

    def process(self, selector: Data, array1: Data, array2: Data, array3: Data, string1: Data, string2: Data, string3: Data):
        if selector is None or selector.data is None:
            return

        selector_value = int(selector.data[0])  # Assume selector is a single-element array
        mode = self.params["settings"]["mode"].value

        if mode == "array":
            if selector_value == 1:
                if array1 is not None:
                    return {"array_out": (array1.data, array1.meta), "string_out": None}
                return
            elif selector_value == 2:
                if array2 is not None:
                    return {"array_out": (array2.data, array2.meta), "string_out": None}
                return
            elif selector_value == 3:
                if array3 is not None:
                    return {"array_out": (array3.data, array3.meta), "string_out": None}
                return
            else:
                raise ValueError("Selector value must be 1 or 2")

        elif mode == "string":
            if selector_value == 1:
                if string1 is not None:
                    return {"array_out": None, "string_out": (string1.data, string1.meta)}
                return
            elif selector_value == 2:
                if string2 is not None:
                    return {"array_out": None, "string_out": (string2.data, string2.meta)}
                return
            elif selector_value == 3:
                if string3 is not None:
                    return {"array_out": None, "string_out": (string3.data, string2.meta)}
                return
            else:
                raise ValueError("Selector value must be 1 or 2")
