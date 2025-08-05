import threading
import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


def hz_to_midi(hz):
    """Convert a frequency in Hz to a MIDI note number."""
    return int(69 + 12 * np.log2(hz / 440.0))


class MidiOut(Node):
    """
    This node sends MIDI note messages to an external MIDI device or software instrument. It receives note numbers (or frequencies), velocities, and durations as input arrays, and sends the corresponding MIDI note on/off messages using the selected output port and channel. Notes can be played either simultaneously or sequentially. The status of the MIDI message transmission is output as a string.

    Inputs:
    - note: An array of MIDI note numbers to play, or frequencies if Hz input is enabled.
    - velocity: An array of velocities for each note. If not provided, default velocity is used.
    - duration: An array of durations in seconds for each note. If not provided, default duration is used.

    Outputs:
    - midi_status: A string indicating whether the MIDI notes were sent successfully, or if there was an error with the note or velocity range.
    """

    def config_input_slots():
        return {"note": DataType.ARRAY, "velocity": DataType.ARRAY, "duration": DataType.ARRAY}

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
                "channel": IntParam(0, 0, 15, doc="MIDI channel"),
                "play_mode": StringParam("simultaneous", options=["simultaneous", "sequential"], doc="Play mode"),
                "default_velocity": IntParam(100, 0, 127, doc="Default MIDI velocity"),
                "default_duration": FloatParam(0.1, 0.001, 10, doc="Default note duration in seconds"),
                "hz_input": BoolParam(False, doc="Interpret input as Hz"),
            }
        }

    def play_note(self, outport, n, v, d, channel):
        """Thread function to play a note."""
        outport.send(self.mido.Message("note_on", note=n, velocity=v, channel=channel))
        time.sleep(d)
        outport.send(self.mido.Message("note_off", note=n, velocity=0, channel=channel))

    def setup(self):
        import mido

        self.mido = mido

    def process(self, note: Data, velocity: Data, duration: Data):
        if note is None or len(note.data) == 0:
            return None

        # Convert Hz to MIDI if required
        midi_notes = [hz_to_midi(n) if self.params.MIDI["hz_input"].value else n for n in note.data]

        # Set up velocities
        if velocity is None:
            velocities = [self.params.MIDI["default_velocity"].value] * len(midi_notes)
        else:
            # Fill in default velocities where necessary
            velocities = [int(v) if v is not None else self.params.MIDI["default_velocity"].value for v in velocity.data]

        # Set up durations
        if duration is None:
            durations = [self.params.MIDI["default_duration"].value] * len(midi_notes)
        else:
            # Fill in default durations where necessary
            durations = [float(d) if d is not None else self.params.MIDI["default_duration"].value for d in duration.data]

        port_name = self.params.MIDI["port_name"].value
        channel = self.params.MIDI["channel"].value
        play_mode = self.params.MIDI["play_mode"].value

        outport = self.mido.open_output(port_name)  # Open the MIDI port once
        alert_on = False
        try:
            if play_mode == "simultaneous":
                threads = []
                for n, v, d in zip(midi_notes, velocities, durations):
                    if n > 127 or n < 0:
                        print("Note outside of MIDI 0-127 range")
                        alert_on = True
                        error_message = "Note outside of MIDI 0-127 range"
                        pass
                    if v > 127 or v < 0:
                        print("Velocity outside of MIDI 0-127 range")
                        alert_on = True
                        error_message = "Velocity outside of MIDI 0-127 range"
                        pass
                    t = threading.Thread(target=self.play_note, args=(outport, n, v, d, channel))
                    t.start()
                    threads.append(t)

                # Wait for all threads to complete
                for t in threads:
                    t.join()
            elif play_mode == "sequential":
                for n, v, d in zip(midi_notes, velocities, durations):
                    if n > 127 or n < 0:
                        print("Note outside of MIDI 0-127 range")
                        alert_on = True
                        error_message = "Note outside of MIDI 0-127 range"
                        pass
                    if v > 127 or v < 0:
                        print("Velocity outside of MIDI 0-127 range")
                        alert_on = True
                        error_message = "Velocity outside of MIDI 0-127 range"
                        pass
                    self.play_note(outport, n, v, d, channel)
        finally:
            outport.close()  # Ensure that the MIDI port is closed when done

        if alert_on:
            return {"midi_status": (f"Notes sent successfully\n{error_message}", note.meta)}
        else:
            return {"midi_status": ("Notes sent successfully", note.meta)}
