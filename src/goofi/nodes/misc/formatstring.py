from goofi.data import DataType, Data
from goofi.node import Node
from goofi.params import StringParam
import re


class FormatString(Node):
    """
    This node combines multiple input strings into a single output string, optionally using a formatting pattern. If no pattern is provided, all input strings are joined together with spaces. If a pattern is given, placeholders within the pattern are replaced with the corresponding input string values. Unnamed placeholders are filled with unused input strings in the order they appear.

    Inputs:
    - input_string_1: Input string to be included in the output.
    - input_string_2: Input string to be included in the output.
    - input_string_3: Input string to be included in the output.
    - input_string_4: Input string to be included in the output.
    - input_string_5: Input string to be included in the output.
    - input_string_6: Input string to be included in the output.
    - input_string_7: Input string to be included in the output.
    - input_string_8: Input string to be included in the output.
    - input_string_9: Input string to be included in the output.
    - input_string_10: Input string to be included in the output.

    Outputs:
    - output_string: The resulting formatted string.
    """

    def config_input_slots():
        slots = {}
        for i in range(1, 11):  # For 10 input strings
            slots[f"input_string_{i}"] = DataType.STRING
        return slots

    def config_output_slots():
        return {"output_string": DataType.STRING}

    def config_params():
        return {
            "pattern": {
                "key": StringParam(
                    "",
                    doc="When empty, will join strings with spaces. You can specify placeholders in the format using curly brackets",
                )
            }
        }

    def process(self, **input_strings):
        pattern = self.params["pattern"]["key"].value

        # Convert Data objects to their string representations
        for key, value in input_strings.items():
            if value is None or value.data is None:
                input_strings[key] = Data(dtype=DataType.STRING, data="", meta={})
            else:
                input_strings[key] = Data(dtype=DataType.STRING, data=value.data, meta=value.meta)

        # Handle the case with no pattern provided
        if not pattern:
            input_values = [value.data for value in input_strings.values()]
            while input_values and not input_values[-1]:
                input_values.pop()
            output = " ".join(input_values)
        else:
            # Replace all named placeholders
            for key, value in input_strings.items():
                pattern = pattern.replace(f"{{{key}}}", value.data)

            # Identify used keys in the pattern (after named replacement to catch any repeating named placeholders)
            used_keys = re.findall(r"{(input_string_\d+)}", pattern)

            # Create a list of unused values
            unused_values = [value.data for key, value in input_strings.items() if key not in used_keys]

            # Replace unnamed placeholders with unused values
            while unused_values and "{}" in pattern:
                pattern = pattern.replace("{}", unused_values.pop(0), 1)

            output = pattern

        return {"output_string": (output, {})}
