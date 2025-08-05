import datetime    
import os    
import time    
import threading    
import queue    
import numpy as np    
from goofi.data import Data, DataType    
from goofi.node import Node    
from goofi.params import StringParam, BoolParam, FloatParam    
    
class SafeSave(Node):      
    def config_input_slots():    
        return {    
            "data": DataType.ARRAY,    
            "start": DataType.ARRAY,     
            "stop": DataType.ARRAY,    
            "fname": DataType.STRING    
        }    
       
    def config_output_slots():    
        return {"status": DataType.STRING}    
        
    def config_params():    
        return {    
            "save": {    
                "filename": StringParam("lsl_data.csv"),    
                "start": BoolParam(False, trigger=True),    
                "stop": BoolParam(False, trigger=True),    
                "duration": FloatParam(0.0, 0.0, 3600.0)    
            }    
        }    
    
    def setup(self):    
        import pandas as pd    
        self.pd = pd    
            
        # Simple unbounded queue    
        self.data_queue = queue.Queue()    
        self.write_thread = None    
        self.stop_event = threading.Event()    
        self.shutdown_complete = threading.Event()  
            
        # Recording state    
        self.is_recording = False    
        self.current_filename = None    
        self.start_time = None    
        self.file_created = False    
    
    def process(self, data: Data, start: Data, stop: Data, fname: Data):    
        # Handle start/stop triggers    
        if (start is not None and (start.data > 0).any()) or self.params.save.start.value:    
            self._start_recording(fname)    
                
        if (stop is not None and (stop.data > 0).any()) or self.params.save.stop.value:    
            self._stop_recording()    
    
        # Clear triggers    
        if self.params.save.start.value:    
            self.params.save.start.value = False    
        if self.params.save.stop.value:    
            self.params.save.stop.value = False    
    
        # Handle duration-based stopping    
        duration = self.params.save.duration.value    
        if self.is_recording and duration > 0:    
            if time.time() - self.start_time > duration:    
                self._stop_recording()    
    
        # Get current queue size for status reporting  
        queue_size = self.data_queue.qsize()  
    
        # Simply append data to queue if recording    
        if self.is_recording and data is not None and data.data is not None:    
            data_item = {    
                'data': data.data.copy(),    
                'meta': data.meta.copy() if data.meta else {},    
                'timestamp': time.time()    
            }    
            self.data_queue.put(data_item)  # Non-blocking put    
            return {"status": (f"recording (queue: {queue_size})", {})}    
            
        return {"status": (f"idle (queue: {queue_size})", {})}    
    
    def _start_recording(self, fname: Data):    
        if self.is_recording:    
            return    
                
        # Generate timestamped filename    
        if fname is not None and fname.data:    
            base_filename = str(fname.data)    
        else:    
            base_filename = self.params.save.filename.value    
                
        basename, ext = os.path.splitext(base_filename)    
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
        self.current_filename = f"{basename}_{datetime_str}.csv"    
            
        # Reset state    
        self.is_recording = True    
        self.start_time = time.time()    
        self.stop_event.clear()    
        self.shutdown_complete.clear()  
        self.file_created = False    
            
        # Start background writer thread (non-daemon for safety)  
        self.write_thread = threading.Thread(    
            target=self._write_worker,     
            daemon=False,  # Non-daemon for guaranteed completion  
            name=f"{self.__class__.__name__}-writer"  
        )    
        self.write_thread.start()    
    
    def _stop_recording(self):    
        if not self.is_recording:    
            return    
                
        self.is_recording = False    
        self.stop_event.set()    
            
        # Wait for writer thread to finish processing all queued data  
        if self.write_thread and self.write_thread.is_alive():    
            # First wait for shutdown to complete gracefully  
            if self.shutdown_complete.wait(timeout=30.0):  
                # Shutdown completed successfully  
                self.write_thread.join(timeout=5.0)  
            else:  
                # Timeout occurred, force join  
                self.write_thread.join(timeout=5.0)  
    
    def _write_worker(self):    
        """Background thread that writes queued data to CSV file"""    
        try:  
            while not self.stop_event.is_set() or not self.data_queue.empty():    
                try:    
                    # Get data from queue    
                    data_item = self.data_queue.get(timeout=0.5)    
                        
                    # Write to CSV file    
                    self._write_to_csv(data_item)    
                        
                except queue.Empty:    
                    continue  
                except Exception as e:  
                    # Log write errors but continue processing  
                    print(f"SafeSave write error: {e}")  
                    continue  
        except Exception as e:  
            print(f"SafeSave worker thread error: {e}")  
        finally:  
            # Signal that shutdown is complete  
            self.shutdown_complete.set()  
    
    def _write_to_csv(self, data_item):    
        """Write single data item to CSV file"""    
        data = data_item['data']    
        timestamp = data_item['timestamp']    
        meta = data_item['meta']    
            
        # Extract channel names from metadata    
        if "channels" in meta and "dim0" in meta["channels"]:    
            channel_names = meta["channels"]["dim0"]    
        else:    
            channel_names = [f"ch_{i}" for i in range(data.shape[0])]    
            
        # Create DataFrame - one row per sample, one column per channel    
        num_samples = data.shape[-1] if data.ndim > 1 else 1    
            
        df_data = {}    
        df_data['timestamp'] = [timestamp] * num_samples    
            
        # Add each channel as a column    
        for i, ch_name in enumerate(channel_names):    
            if i < data.shape[0]:    
                if data.ndim == 1:    
                    df_data[ch_name] = [data[i]]    
                else:    
                    df_data[ch_name] = data[i, :].tolist()    
                    
        df = self.pd.DataFrame(df_data)    
            
        # Write to file (append mode, header only on first write)    
        write_header = not self.file_created    
        df.to_csv(self.current_filename, mode='a', header=write_header, index=False)    
        self.file_created = True    
    
    def terminate(self):    
        if self.is_recording:    
            self._stop_recording()