import datetime
import os
import queue
import threading
import time
import warnings
from pathlib import Path

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


class WriteCsvSafe(Node):
    def config_input_slots():
        return {
            "data": DataType.ARRAY,
            "annot": DataType.TABLE,
            "start": DataType.ARRAY,
            "stop": DataType.ARRAY,
            "fname": DataType.STRING,
        }

    def config_output_slots():
        return {"status": DataType.STRING}

    def config_params():
        return {
            "save": {
                "filename": StringParam("lsl_data.csv"),
                "start": BoolParam(False, trigger=True),
                "stop": BoolParam(False, trigger=True),
                "duration": FloatParam(0.0, 0.0, 3600.0, doc="Maximum recording duration in seconds"),
            },
            "common": {
                "autotrigger": True,
                "max_frequency": 60,  # Run this node faster to not risk losing any data packets
            },
        }

    def setup(self):
        import pandas as pd

        self.pd = pd

        # Simple unbounded queue
        self.data_queue = queue.Queue()
        self.write_thread = None
        self.stop_event = threading.Event()

        # Recording state
        self.is_recording = False
        self.current_filename = None
        self.start_time = None
        self.file_created = False
        self.columns = None

    def process(self, data: Data, annot: Data, start: Data, stop: Data, fname: Data):
        # NOTE: We need to clear input slots because this node has autotrigger=True and a high max_frequency,
        # if we didn't clear the input slots we would likely repeat data packets in the output file. We do not
        # clear annot or fname as these should be repeated for subsequent entries if they didn't change.
        self.input_slots["data"].clear()
        self.input_slots["start"].clear()
        self.input_slots["stop"].clear()

        # handle start/stop triggers
        if (start is not None and (start.data > 0).any()) or self.params.save.start.value:
            self._start_recording(fname)

        if (stop is not None and (stop.data > 0).any()) or self.params.save.stop.value:
            self._stop_recording()

        # handle duration-based stopping
        duration = self.params.save.duration.value
        if self.is_recording and duration > 0:
            if time.time() - self.start_time > duration:
                self._stop_recording()

        # get current queue size for status reporting
        queue_size = self.data_queue.qsize()

        # append data and annot to queue if recording
        if self.is_recording and data is not None:
            self.data_queue.put((data, annot), block=False)

        if self.is_recording:
            return {"status": (f"recording (queue: {queue_size})", {})}
        return {"status": (f"idle (queue: {queue_size})", {})}

    def _start_recording(self, fname: Data):
        if self.is_recording:
            return

        # add timestamp to filename
        if fname is not None and fname.data:
            base_filename = str(fname.data)
        else:
            base_filename = self.params.save.filename.value

        basename = os.path.splitext(base_filename)[0]
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_filename = f"{basename}_{datetime_str}.csv"
        Path(self.current_filename).parent.mkdir(parents=True, exist_ok=True)

        # reset state
        self.is_recording = True
        self.start_time = time.time()
        self.stop_event.clear()
        self.file_created = False

        # start background write thread (non-daemon to complete write after shutdown)
        self.write_thread = threading.Thread(target=self._write_worker, daemon=False)
        self.write_thread.start()

    def _stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.stop_event.set()

        # clear annot input slot to avoid keeping old data
        self.input_slots["annot"].clear()

        # wait for write thread to finish processing all queued data
        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=5.0)

    def _write_worker(self):
        """Background thread that writes queued data to CSV file"""
        try:
            while not self.stop_event.is_set() or not self.data_queue.empty():
                try:
                    # retrieve data packet and write to csv
                    data_item, annot = self.data_queue.get(timeout=0.5)
                    self._write_to_csv(data_item, annot)
                except queue.Empty:
                    continue
                except Exception as e:
                    # log write errors but continue processing
                    print(f"WriteCsvSafe write error: {e}")
                    continue
        except Exception as e:
            print(f"WriteCsvSafe worker thread error: {e}")

    def _write_to_csv(self, data_item: Data, annot: Data):
        """Write single data item with annotations to CSV file"""
        data = data_item.data
        meta = data_item.meta

        # Extract channel names from metadata
        if "channels" in meta and "dim0" in meta["channels"]:
            channel_names = meta["channels"]["dim0"]
        else:
            channel_names = [f"ch_{i}" for i in range(data.shape[0])]

        num_samples = 1 if data.ndim == 1 else data.shape[1]

        if self.columns is None:
            self.columns = channel_names.copy()
            if annot is not None:
                self.columns += list(annot.data.keys())
                if len(self.columns) != len(set(self.columns)):
                    warnings.warn("Duplicate column names detected, ignoring duplicates.")

            # add a special column to group extra annotations in (those that were not present during file creation)
            # must be the last column!
            self.columns += ["_extra_annot"]

        # write data and length-matched annotations into the data frame dict
        df_data = {}
        for col in self.columns:
            if col in channel_names:
                data_idx = channel_names.index(col)
                if data.ndim == 1:
                    df_data[col] = [data[data_idx]]
                else:
                    df_data[col] = data[data_idx].tolist()
            elif col in annot.data:
                item = annot.data.pop(col).data
                if isinstance(item, np.ndarray) and item.size == 1:
                    item = item.item()
                df_data[col] = [item] * num_samples
            elif col != "_extra_annot":
                df_data[col] = [None] * num_samples

        # add extra annotations
        extra_annots = {k: v.data for k, v in annot.data.items()}
        df_data["_extra_annot"] = [extra_annots] * num_samples

        df = self.pd.DataFrame(df_data)

        # Write to file (append mode, header only on first write)
        write_header = not self.file_created
        df.to_csv(self.current_filename, mode="a", header=write_header, index=False)
        self.file_created = True

    def terminate(self):
        if self.is_recording:
            self._stop_recording()
