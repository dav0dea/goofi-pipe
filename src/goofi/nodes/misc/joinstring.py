from goofi.data import DataType
from goofi.node import Node


class JoinString(Node):
    """
    Joins up to five input strings into a single string, using a configurable separator. Only selected and non-empty inputs are included in the concatenation.

    Inputs:
    - string1: The first string to join.
    - string2: The second string to join.
    - string3: The third string to join.
    - string4: The fourth string to join.
    - string5: The fifth string to join.

    Outputs:
    - output: The concatenated string result.
    """

    def config_input_slots():
        return {
            "string1": DataType.STRING,
            "string2": DataType.STRING,
            "string3": DataType.STRING,
            "string4": DataType.STRING,
            "string5": DataType.STRING,
        }

    def config_output_slots():
        return {"output": DataType.STRING}

    def config_params():
        return {
            "join": {
                "separator": ", ",
                "string1": True,
                "string2": True,
                "string3": True,
                "string4": True,
                "string5": True,
            }
        }

    def process(self, **inputs):
        separator = self.params.join.separator.value
        selected_inputs = filter(lambda item: self.params.join[item[0]].value and item[1] is not None, inputs.items())
        output = separator.join([item[1].data for item in selected_inputs])
        return {"output": (output, {})}
