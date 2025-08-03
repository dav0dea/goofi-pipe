import threading

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class MidiCCout(Node):
    """
    This node converts up to five arrays of input values into MIDI Control Change (CC) messages and sends them out to a selected MIDI output port. Each input array corresponds to a configurable MIDI CC number, allowing for real-time control of MIDI devices or software via multiple CC messages. The node provides status output indicating the success or any errors during message transmission.

    Inputs:
    - cc1: Array of values to send as MIDI CC messages using the first configured CC number.
    - cc2: Array of values to send as MIDI CC messages using the second configured CC number.
    - cc3: Array of values to send as MIDI CC messages using the third configured CC number.
    - cc4: Array of values to send as MIDI CC messages using the fourth configured CC number.
    - cc5: Array of values to send as MIDI CC messages using the fifth configured CC number.

    Outputs:
    - midi_status: Status message indicating whether the MIDI CC messages were sent successfully or if errors occurred.
    """

    def config_input_slots():
        return {
            "cc1": DataType.ARRAY,
            "cc2": DataType.ARRAY,
            "cc3": DataType.ARRAY,
            "cc4": DataType.ARRAY,
            "cc5": DataType.ARRAY,
        }

    def config_output_slots():
        return {"midi_status": DataType.STRING}

    def config_params():
        try:
            import mido

            available_ports = mido.get_output_names()
        except Exception as e:
            print(f"Error getting MIDI ports: {e}")
            available_ports = []

        return {
            "MIDI": {
                "port_name": StringParam("", options=available_ports, doc="MIDI output port name"),
                "channel": IntParam(1, 1, 15, doc="MIDI channel"),
                "cc1": IntParam(1, 0, 127, doc="MIDI CC number for input 1"),
                "cc2": IntParam(2, 0, 127, doc="MIDI CC number for input 2"),
                "cc3": IntParam(3, 0, 127, doc="MIDI CC number for input 3"),
                "cc4": IntParam(4, 0, 127, doc="MIDI CC number for input 4"),
                "cc5": IntParam(5, 0, 127, doc="MIDI CC number for input 5"),
            }
        }

    def send_cc(self, outport, cc_number, value, channel):
        """Send a MIDI CC message."""
        # convert cc_number to int
        cc_number = int(cc_number)
        value = int(value)
        outport.send(self.mido.Message("control_change", control=cc_number, value=value, channel=channel))

    def setup(self):
        import mido

        self.mido = mido

    def process(self, cc1: Data, cc2: Data, cc3: Data, cc4: Data, cc5: Data):
        port_name = self.params.MIDI["port_name"].value
        channel = self.params.MIDI["channel"].value
        channel = channel - 1  # MIDI channels are 0-indexed
        cc_numbers = [
            self.params.MIDI["cc1"].value,
            self.params.MIDI["cc2"].value,
            self.params.MIDI["cc3"].value,
            self.params.MIDI["cc4"].value,
            self.params.MIDI["cc5"].value,
        ]
        cc_data = [cc1, cc2, cc3, cc4, cc5]

        outport = self.mido.open_output(port_name) if port_name != "goofi" else self.goofi_port

        alert_on = False
        try:
            threads = []
            for i, data in enumerate(cc_data):
                if data is not None and len(data.data) > 0:
                    for value in data.data:
                        if value < 0 or value > 127:
                            print(f"Value outside of MIDI 0-127 range: {value}")
                            alert_on = True
                            error_message = f"Value outside of MIDI 0-127 range: {value}"
                            pass
                        t = threading.Thread(target=self.send_cc, args=(outport, cc_numbers[i], value, channel))
                        t.start()
                        threads.append(t)

            for t in threads:
                t.join()
        finally:
            if port_name != "goofi":
                outport.close()  # Ensure that the MIDI port is closed when done

        if alert_on:
            return {"midi_status": (f"CC messages sent with errors\n{error_message}", cc1.meta)}
        else:
            return {"midi_status": ("CC messages sent successfully", cc1.meta)}
