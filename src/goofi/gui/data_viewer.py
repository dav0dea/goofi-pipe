import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np
from mne import channels
from scipy.interpolate import griddata

from goofi.data import Data, DataType
from goofi.gui.events import is_ctrl_down, is_shift_down
from goofi.message import Message, MessageType


class ViewerContainer:
    def __init__(
        self,
        dtype: DataType,
        content_window: int,
        collapsed: bool,
        sibling_containers: Dict[str, "ViewerContainer"],
        log_scale_x: bool = False,
        log_scale_y: bool = False,
        viewer_idx: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.dtype = dtype
        self.content_window = content_window
        self.sibling_containers = sibling_containers

        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.collapsed = collapsed

        self.viewer_idx = viewer_idx
        self.viewer = DTYPE_VIEWER_MAP[self.dtype][self.viewer_idx](self.content_window, self)
        self.viewer.set_size()

        # initialize axis scaling
        self.update_axis_scaling()

        # set viewer size if provided
        min_size = dpg.get_item_user_data(self.content_window)[0]
        self.width = width or min_size[0]
        self.height = height or min_size[1]
        self.set_size()

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=0, callback=self.clicked)
            dpg.add_mouse_wheel_handler(callback=self.wheel)

    def clicked(self, _1: int, _2: Any) -> None:
        """Content window click handler. If the content window is ctrl+clicked, the viewer is switched."""
        try:
            state = dpg.get_item_state(self.content_window)
        except SystemError:
            # failed to retrieve state, likely because the window was closed
            return

        if "hovered" not in state or not state["hovered"]:
            # window is not hovered
            return

        if is_ctrl_down():
            # ctrl+click: switch to next viewer
            self.next_viewer()

    def wheel(self, _1: int, value: Any) -> None:
        """Content window mouse wheel handler. Increases or deacreases the size of the viewer."""
        try:
            state = dpg.get_item_state(self.content_window)
        except SystemError:
            # failed to retrieve state, likely because the window was closed
            return

        if "hovered" not in state or not state["hovered"]:
            # window is not hovered
            return

        # increase or decrease size of viewer
        min_size = dpg.get_item_user_data(self.content_window)[0]
        self.width = max(min_size[0], self.width + value * 10)
        if not is_shift_down():
            self.height = max(min_size[1], self.height + value * 10)

        # apply changes
        self.set_size()

    def set_size(self, new_width: Optional[int] = None, new_height: Optional[int] = None, propagate: bool = True) -> None:
        """Set the size of the viewer."""
        new_width = new_width or self.width
        new_height = new_height or self.height

        # adjust width to always span the available width based on siblings
        if len(self.sibling_containers) > 1:
            new_width = max(new_width, max([sib.width for sib in self.sibling_containers.values() if self != sib]))
            if propagate:
                for sib in self.sibling_containers.values():
                    if self == sib:
                        continue
                    sib.set_size(new_width=new_width, propagate=False)

        if self.collapsed:
            new_height = 0

        dpg.set_item_width(self.content_window, new_width)
        dpg.set_item_height(self.content_window, new_height)

        header = dpg.get_item_parent(self.content_window)
        window = dpg.get_item_parent(header)

        header_height = dpg.get_item_state(header)["rect_size"][1]
        if header_height == 0:
            # this happens when the header is off screen, set to default height
            header_height = 15

        dpg.set_item_width(window, new_width)
        dpg.set_item_height(window, new_height + header_height + 4)  # magic + 4 to avoid scroll bar

        self.viewer.set_size()

    def next_viewer(self) -> None:
        """Switch to the next viewer."""
        dpg.delete_item(self.content_window, children_only=True)
        self.viewer_idx = (self.viewer_idx + 1) % len(DTYPE_VIEWER_MAP[self.dtype])
        self.viewer = DTYPE_VIEWER_MAP[self.dtype][self.viewer_idx](self.content_window, self)
        self.viewer.set_size()
        self.update_axis_scaling()

    def toggle_log_plot(self, axis: str) -> None:
        """Toggle log scale on the specified axis."""
        if axis == "x":
            self.log_scale_x = not self.log_scale_x
        elif axis == "y":
            self.log_scale_y = not self.log_scale_y
        else:
            raise ValueError(f"Invalid axis {axis}.")

        # apply changes
        self.update_axis_scaling()

    def update_axis_scaling(self) -> None:
        """Update the axis scaling."""
        if hasattr(self.viewer, "xax"):
            dpg.configure_item(self.viewer.xax, log_scale=self.log_scale_x)
        if hasattr(self.viewer, "yax"):
            dpg.configure_item(self.viewer.yax, log_scale=self.log_scale_y)

    def get_state(self) -> dict:
        return {
            "viewer_idx": self.viewer_idx,
            "width": self.width,
            "height": self.height,
            "collapsed": self.collapsed,
            "log_scale_x": self.log_scale_x,
            "log_scale_y": self.log_scale_y,
        }

    def __call__(self, msg: Message) -> Any:
        if not msg.type == MessageType.DATA:
            raise ValueError(f"Expected message type DATA, got {msg.type}.")
        if not msg.content["data"].dtype == self.dtype:
            raise ValueError(f"Expected data type {self.dtype}, got {msg.content['data'].dtype}.")

        try:
            # update the data viewer
            self.viewer.update(msg.content["data"])
        except UnsupportedViewerError:
            self.next_viewer()
        except Exception:
            traceback.print_exc()
            print(f"Error while updating data viewer for data type {self.dtype}.")


class DataViewer(ABC):
    def __init__(self, dtype: DataType, content_window: int, container: ViewerContainer) -> None:
        self.dtype = dtype
        self.content_window = content_window
        self.container = container

    @abstractmethod
    def update(self, data: Data) -> None:
        pass

    def set_size(self) -> None:
        pass


class ArrayViewer(DataViewer):
    def __init__(self, content_window: int, container: ViewerContainer) -> None:
        super().__init__(DataType.ARRAY, content_window, container)

        self.vmin = None
        self.vmax = None
        self.margin = 0.1
        self.shrinking = 0.01

        size = dpg.get_item_user_data(content_window)[0]
        if size is None:
            raise RuntimeError("Expected content window to have user data with size.")

        # create plot
        self.plot = dpg.add_plot(
            width=size[0],
            height=size[1],
            parent=self.content_window,
            no_menus=True,
            no_box_select=True,
            no_mouse_pos=True,
            no_highlight=True,
            no_child=True,
        )

        # add x and y axis
        axis_kwargs = dict(parent=self.plot, no_gridlines=True, no_tick_marks=True, lock_min=True, lock_max=True)
        self.xax = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, **axis_kwargs)
        self.yax = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=False, **axis_kwargs)

        # container for DPG line series
        self.line_series = []

    def update(self, data: Data) -> None:
        """
        This function handles drawing numerical data to a plot. Array shapes are handled as follows:
        - 0D (single number): the data is drawn as a horizontal line.
        - 1D: the data is drawn as a simple line plot.
        - 2D: the array gets interpreted as (n_channels, n_samples), and each channel is drawn as a line plot.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if self.vmin is not None and self.vmax is not None:
            # apply shrinking to vmin and vmax
            self.vmin = self.vmin * (1 - self.shrinking) + self.vmax * self.shrinking
            self.vmax = self.vmax * (1 - self.shrinking) + self.vmin * self.shrinking

        # update vmin and vmax
        if self.vmin is None or np.min(array) < self.vmin:
            self.vmin = np.nanmin(array)
        if self.vmax is None or np.max(array) > self.vmax:
            self.vmax = np.nanmax(array)

        if array.size == 1:
            ##########################
            # 0D: draw vertical line #
            ##########################

            # set x and y-axis ticks
            dpg.configure_item(self.xax, no_tick_labels=False)
            dpg.configure_item(self.yax, no_tick_labels=True)

            # remove extra data series
            while len(self.line_series) > 1:
                dpg.delete_item(self.line_series.pop())

            if len(self.line_series) == 0:
                # add new data series
                self.line_series.append(dpg.add_line_series([array, array], [0, 1], parent=self.yax))
            else:
                # update existing data series
                dpg.set_value(self.line_series[0], [[array, array], [0, 1]])

            # autoscale x and y-axis limits
            dpg.set_axis_limits(self.xax, self.vmin - abs(self.vmax) * self.margin, self.vmax + abs(self.vmax) * self.margin)
            dpg.set_axis_limits(self.yax, 0, 1)
        else:
            ############################
            # 1D or 2D: draw line plot #
            ############################

            # set x and y-axis ticks
            dpg.configure_item(self.xax, no_tick_labels=True)
            dpg.configure_item(self.yax, no_tick_labels=False)

            if array.ndim == 1:
                # remove extra data series
                while len(self.line_series) > 1:
                    dpg.delete_item(self.line_series.pop())

                # TODO: use dim0 numerical labels if present
                xs = np.arange(array.shape[0]) + 1

                if len(self.line_series) == 0:
                    # add new data series
                    self.line_series.append(dpg.add_line_series(xs, array, parent=self.yax))
                else:
                    # update existing data series
                    dpg.set_value(self.line_series[0], [xs, array])
            elif array.ndim == 2:
                # remove extra data series
                while len(self.line_series) > array.shape[0]:
                    dpg.delete_item(self.line_series.pop())

                # TODO: use dim0 numerical labels if present
                xs = np.arange(array.shape[1]) + 1

                # iterate over channels (first dimension)
                for i in range(array.shape[0]):
                    if len(self.line_series) == i:
                        # add new data series
                        self.line_series.append(dpg.add_line_series(xs, array[i], parent=self.yax))
                    else:
                        # update existing data series
                        dpg.set_value(self.line_series[i], [xs, array[i]])
            else:
                raise UnsupportedViewerError(f"Cannot handle array with {array.ndim} dimensions.")

            # autoscale x and y-axis limits
            dpg.set_axis_limits(self.xax, xs.min(), xs.max())
            m = abs(self.vmax - self.vmin) * self.margin
            if self.container.log_scale_y:
                # TODO: can we have margins with log scale?
                dpg.set_axis_limits(self.yax, self.vmin, self.vmax)
            else:
                dpg.set_axis_limits(self.yax, self.vmin - m, self.vmax + m)

    def set_size(self) -> None:
        """This function sets the size of the plot."""
        dpg.set_item_width(self.plot, dpg.get_item_width(self.content_window))
        dpg.set_item_height(self.plot, dpg.get_item_height(self.content_window))


class ImageViewer(DataViewer):
    def __init__(self, content_window: int, container: ViewerContainer, max_res: int = 512) -> None:
        super().__init__(DataType.ARRAY, content_window, container)

        config = dpg.get_item_configuration(content_window)
        res = (config["width"], config["height"])

        # set resolution to max_res
        if res[0] != max_res and res[1] != max_res:
            scale = max_res / max(res)
            res = (int(res[0] * scale), int(res[1] * scale))

        # initialize texture
        with dpg.texture_registry():
            self.texture = dpg.add_dynamic_texture(width=res[0], height=res[1], default_value=[0.0 for _ in range(res[0] * res[1] * 4)])
        self.image = dpg.add_image(self.texture, parent=self.content_window)
        self.res = res

    def update(self, data: Data) -> None:
        """
        This function handles drawing image data to an image item.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if array.ndim > 3:
            raise UnsupportedViewerError(f"Cannot handle array with {array.ndim} dimensions.")
        while array.ndim < 3:
            array = array[..., None]

        # make sure we have 4 channels
        if array.shape[2] > 4:
            raise NotImplementedError(f"Cannot handle array with {array.shape[2]} channels.")
        elif array.shape[2] == 3:
            array = np.concatenate([array, np.ones((*array.shape[:2], 1))], axis=2)
        elif array.shape[2] == 2:
            array = np.concatenate([array, np.ones((*array.shape[:2], 2))], axis=2)
        elif array.shape[2] == 1:
            array = np.concatenate([array] * 3 + [np.ones((*array.shape[:2], 1))], axis=2)

        # clamp values to [0, 1]
        array = np.clip(array, 0, 1)

        tex_config = dpg.get_item_configuration(self.texture)
        if tex_config["height"] != array.shape[1] or tex_config["width"] != array.shape[0]:
            # resize array to fit texture
            array = cv2.resize(array, (tex_config["width"], tex_config["height"]), interpolation=cv2.INTER_NEAREST)

        # update texture
        dpg.set_value(self.texture, array.flatten())

    def set_size(self) -> None:
        """This function sets the size of the image."""
        dpg.set_item_width(self.image, dpg.get_item_width(self.content_window))
        dpg.set_item_height(self.image, dpg.get_item_height(self.content_window))


class TrajectoryViewer(ArrayViewer):
    def update(self, data: Data) -> None:
        """
        This function handles drawing numerical trajectories to a plot. Trajectories are all
        (n x frames) arrays,
        where the first row is the x-axis and the second row is the y-axis.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if self.vmin is not None and self.vmax is not None:
            # apply shrinking to vmin and vmax
            self.vmin = self.vmin * (1 - self.shrinking) + self.vmax * self.shrinking
            self.vmax = self.vmax * (1 - self.shrinking) + self.vmin * self.shrinking

        # update vmin and vmax
        if self.vmin is None or np.min(array) < self.vmin:
            self.vmin = np.nanmin(array)
        if self.vmax is None or np.max(array) > self.vmax:
            self.vmax = np.nanmax(array)

        if array.ndim != 2 or array.shape[0] < 2:
            raise UnsupportedViewerError(f"Cannot handle array with shape {array.shape}.")

        # set x and y-axis ticks
        dpg.configure_item(self.xax, no_tick_labels=False)
        dpg.configure_item(self.yax, no_tick_labels=False)

        # Gauss Sum to calculate the number of trajectories
        n = array.shape[0]
        n = n * (n - 1) / 2

        # remove extra data series
        while len(self.line_series) > n:
            dpg.delete_item(self.line_series.pop())

        while len(self.line_series) < n:
            # add new data series
            self.line_series.append(dpg.add_line_series([0, 1], [0, 1], parent=self.yax))

        idx = 0
        for i in range(array.shape[0]):
            for j in range(i + 1, array.shape[0]):
                # update existing data series
                xs, ys = array[i], array[j]
                dpg.set_value(self.line_series[idx], [xs, ys])
                idx += 1

        # autoscale x and y-axis limits
        dpg.set_axis_limits(self.xax, self.vmin - abs(self.vmax) * self.margin, self.vmax + abs(self.vmax) * self.margin)
        dpg.set_axis_limits(self.yax, self.vmin - abs(self.vmax) * self.margin, self.vmax + abs(self.vmax) * self.margin)


class TopomapViewer(ImageViewer):
    def __init__(self, content_window: int, container: ViewerContainer) -> None:
        super().__init__(content_window, container)

        self.layout = channels.read_layout("EEG1005")
        self.cmap = plt.get_cmap("inferno")

    def update(self, data: Data) -> None:
        """
        This function handles drawing image data to an image item.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if array.ndim > 1:
            raise UnsupportedViewerError("Cannot handle array with more than one dimension.")

        if "dim0" not in data.meta["channels"]:
            raise UnsupportedViewerError("Expected data to have channel dim0.")

        # get channel positions
        idxs, pos = [], []
        for i, ch in enumerate(data.meta["channels"]["dim0"]):
            if ch in self.layout.names:
                idxs.append(i)
                pos.append(self.layout.pos[self.layout.names.index(ch), :2])

        if len(idxs) == 0:
            raise UnsupportedViewerError("No channels found in layout.")

        # create image
        vmin = np.nanmin([p[0] for p in pos] + [0]), np.nanmin([p[1] for p in pos] + [0])
        vmax = np.nanmax([p[0] for p in pos] + [1]), np.nanmax([p[1] for p in pos] + [1])

        xs = np.linspace(vmin[0], vmax[0], self.res[0])
        ys = np.linspace(vmax[1], vmin[1], self.res[1])
        grid = np.stack(np.meshgrid(xs, ys), axis=-1)
        grid = grid.reshape(-1, 2)

        img = griddata(pos, array[idxs], grid, method="cubic", fill_value=0)
        img = img.reshape(self.res)
        img = self.cmap(img)

        tex_config = dpg.get_item_configuration(self.texture)
        if tex_config["height"] != img.shape[1] or tex_config["width"] != img.shape[0]:
            # resize array to fit texture
            img = cv2.resize(img, (tex_config["width"], tex_config["height"]), interpolation=cv2.INTER_NEAREST)

        # update texture
        dpg.set_value(self.texture, img.flatten())

    def set_size(self) -> None:
        """This function sets the size of the image."""
        dpg.set_item_width(self.image, dpg.get_item_width(self.content_window))
        dpg.set_item_height(self.image, dpg.get_item_height(self.content_window))


class StringViewer(DataViewer):
    def __init__(self, content_window: int, container: ViewerContainer) -> None:
        super().__init__(DataType.STRING, content_window, container)

        # create text item
        self.text = dpg.add_text("", parent=self.content_window, wrap=dpg.get_item_width(self.content_window))

    def update(self, data: Data) -> None:
        """
        This function handles drawing string data to a text item.

        ### Parameters
        `data` : Data
            The data message.
        """
        dpg.set_value(self.text, data.data)

    def set_size(self) -> None:
        """This function sets the size of the text item."""
        dpg.configure_item(self.text, wrap=dpg.get_item_width(self.content_window))


class TableViewer(DataViewer):
    def __init__(self, content_window: int, container: ViewerContainer) -> None:
        super().__init__(DataType.TABLE, content_window, container)

        self.width = dpg.get_item_width(self.content_window)

        # create table
        self.table = dpg.add_table(parent=self.content_window, header_row=False, policy=dpg.mvTable_SizingFixedFit)
        dpg.add_table_column(parent=self.table)
        dpg.add_table_column(parent=self.table)

        # row container
        self.rows = []

    def update(self, data: Data) -> None:
        """
        This function handles drawing table data to a table item.

        ### Parameters
        `data` : Data
            The data message.
        """
        # remove extra rows
        while len(self.rows) > len(data.data):
            dpg.delete_item(self.rows.pop())
        # add missing rows
        while len(self.rows) < len(data.data):
            with dpg.table_row(parent=self.table) as row:
                key = dpg.add_text("")
                val = dpg.add_text("")
                dpg.set_item_user_data(row, (key, val))
                # store row to populate later
                self.rows.append(row)

        # populate rows with data
        for row, (key, val) in zip(self.rows, data.data.items()):
            # truncate value text
            val = str(val.data).replace("\n", " ")
            cut = int(self.width / 7 - 6)
            if len(val) > cut:
                val = val[: cut - 3] + "..."

            # update key and value cells
            key_cell, val_cell = dpg.get_item_user_data(row)
            dpg.set_value(key_cell, key)
            dpg.set_value(val_cell, val)

    def set_size(self) -> None:
        """This function sets the size of the table."""
        self.width = dpg.get_item_width(self.content_window)


DTYPE_VIEWER_MAP = {
    DataType.ARRAY: [ArrayViewer, ImageViewer, TrajectoryViewer, TopomapViewer],
    DataType.STRING: [StringViewer],
    DataType.TABLE: [TableViewer],
}


class UnsupportedViewerError(Exception):
    pass
