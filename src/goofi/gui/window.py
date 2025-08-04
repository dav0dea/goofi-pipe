import importlib.resources as pkg_resources
import os
import platform
import pprint
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from goofi import assets
from goofi.data import DataType
from goofi.gui import events
from goofi.gui.data_viewer import ViewerContainer
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef, list_nodes
from goofi.params import BoolParam, FloatParam, IntParam, Param, StringParam

DTYPE_SHAPE_MAP = {
    DataType.ARRAY: dpg.mvNode_PinShape_CircleFilled,
    DataType.STRING: dpg.mvNode_PinShape_TriangleFilled,
    DataType.TABLE: dpg.mvNode_PinShape_QuadFilled,
}


NODE_CAT_COLORS = [
    [110, 88, 208],
    [129, 178, 154],
    [242, 204, 143],
    [102, 168, 212],
    [224, 122, 95],
    [178, 144, 205],
]

PARAM_WINDOW_WIDTH = 400


class MetadataPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if isinstance(object, float):
            return (f"{object:.1f}", True, False)
        return super().format(object, context, maxlevels, level)


data = {
    "apple": 4.123456,
    "banana": 3.987654,
    "fruits": [{"name": "cherry", "value": 2.1234}, {"name": "blueberry", "value": 1.456789}],
}


def format_name(name: str) -> str:
    """Format a name to be used as a node label."""
    return name.replace("_", " ").title()


def running(func):
    """Decorator to make sure the window is running before calling a function."""

    def wrapper(*args, **kwargs):
        # make sure DearPyGui is running
        n_tries = 5
        while not dpg.is_dearpygui_running() and n_tries > 0:
            time.sleep(0.1)
            n_tries -= 1
        if n_tries == 0:
            raise RuntimeError("DearPyGui is not running.")
        return func(*args, **kwargs)

    return wrapper


def update_minimap(func):
    """Decorator to update the minimap state after the node editor is updated."""

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

        # update minimap state
        if len(args[0].nodes) > 0:
            dpg.configure_item(args[0].node_editor, minimap=True)
        else:
            dpg.configure_item(args[0].node_editor, minimap=False)

    return wrapper


@dataclass
class GUINode:
    item: int
    input_slots: Dict[str, int]
    output_slots: Dict[str, int]
    output_draw_handlers: Dict[str, int]
    node_ref: NodeRef
    _error_msg: str = None

    def __post_init__(self):
        self.set_error(None, Window())

    def set_error(self, msg: str, win: "Window") -> None:
        """Set the error message."""
        # TODO: revert back to the standard node categories (without _empty suffix) once https://github.com/hoffstadt/DearPyGui/issues/2444 is fixed
        theme_name = self.node_ref.category
        theme_name += "_empty" if len(self.input_slots) == 0 and len(self.output_slots) == 0 else ""

        if msg is None:
            # no error, reset theme
            try:
                dpg.bind_item_theme(self.item, win.node_themes[theme_name])
            except SystemError:
                # the node might have been deleted, ignore this error
                pass
            return

        if msg == self._error_msg:
            # nothing changed
            return

        self._error_msg = msg

        # we have an error, set theme
        dpg.bind_item_theme(self.item, win.node_error_themes[theme_name])

        # update the info window if it is open
        info_win = win.node_info_window
        if info_win is not None and dpg.get_item_user_data(info_win) == self.item:
            dpg.set_value(dpg.get_item_children(info_win)[1][2], msg)

    def display_info(self, win: "Window") -> None:
        """Display information about the node."""
        if win.node_info_window is not None:
            # close the current info window
            dpg.delete_item(win.node_info_window)
            win.node_info_window = None

        # create new info window
        size = (500, 300)
        pos = dpg.get_mouse_pos(local=False)
        pos = (pos[0] - size[0] / 2, pos[1] - size[1] / 2)

        with dpg.theme() as win_theme:
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [200, 200, 200, 255], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, [0, 0, 0, 255], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)

        with dpg.window(
            width=size[0],
            height=size[1],
            pos=pos,
            no_move=True,
            no_close=True,
            no_collapse=True,
            no_resize=True,
            no_title_bar=True,
            user_data=self.item,
        ) as win.node_info_window:
            dpg.add_text("Node Info")
            dpg.add_separator()

            txt = "All good!" if self._error_msg is None else self._error_msg
            dpg.add_input_text(multiline=True, default_value=txt.strip(), readonly=True, width=-1, height=-1)

            dpg.bind_item_theme(win.node_info_window, win_theme)


def handle_data(win: "Window", gui_node: GUINode, node: NodeRef, message: Message):
    """
    Handle a data message from a node. This function is registered as a message handler for the
    `MessageType.DATA` message type.

    ### Parameters
    `gui_node` : GUINode
        The GUI node instance.
    `node` : NodeRef
        The node reference.
    `message` : Message
        The data message.
    """
    try:
        gui_node.output_draw_handlers[message.content["slot_name"]](message)
        gui_node.set_error(None, win)
    except ValueError as e:
        # TODO: add proper logging
        print(f"Output draw handler for slot {message.content['slot_name']} failed: {e}")
    except KeyError:
        print(gui_node, message.content["slot_name"], gui_node.output_draw_handlers.keys())
    except Exception as e:
        print(f"Error in output draw handler for slot {message.content['slot_name']}: {e}")

    if win.metadata_view is not None and win.selected_node == gui_node.item:
        try:
            dpg.set_value(
                win.metadata_view[message.content["slot_name"]],
                MetadataPrinter(compact=True, width=50).pformat(message.content["data"].meta),
            )
        except SystemError:
            # param window was closed, ignore this error
            pass


def toggle_log_plot(_1, _2, data):
    """Toggle the log plot state of a data viewer."""
    _, node, slot, axis = data
    node.output_draw_handlers[slot].toggle_log_plot(axis)


def param_updated(a, value, user_data):
    """
    A GUI callback function for updating a parameter value. This is called by DearPyGui when
    a parameter is updated, and passed the update to the node.
    """
    group, name, node = user_data[:3]

    if len(user_data) == 4:
        # callback from a trigger button, value should be True
        value = user_data[3]
    elif len(user_data) == 5:
        # the parameter includes multiple input widgets, update all of them
        input_group, value_type = user_data[3:]

        try:
            # make sure the value has the correct type
            value = value_type(value)
        except ValueError:
            # failed to convert value to correct type, ignore this update
            return

        # update all input widgets
        for child in dpg.get_item_children(input_group)[1]:
            try:
                dpg.set_value(child, value)
            except SystemError:
                # the input widget might have been deleted, ignore this error
                pass

    if value == node.params[group][name].value:
        # value did not change, ignore this update
        return

    # send the updated parameter to the node
    node.update_param(group, name, value)
    # mark manager state as dirty
    Window().manager.unsaved_changes = True


def add_param(parent: int, group: str, name: str, param: Param, node: NodeRef) -> None:
    """
    Add a parameter to the GUI.

    ### Parameters
    `parent` : int
        The parent item.
    `group` : str
        The parameter group.
    `name` : str
        The parameter name.
    `param` : Param
        The parameter.
    `node` : NodeRef
        The node reference.
    """
    with dpg.table_row(parent=parent):
        with dpg.table_cell():
            lbl = dpg.add_text(format_name(name))
            if param.doc is not None:
                dpg.add_text(param.doc, parent=dpg.add_tooltip(parent=lbl))

        with dpg.table_cell():
            if isinstance(param, BoolParam):
                # parameter is a bool
                if param.trigger:
                    dpg.add_button(label=format_name(name), callback=param_updated, user_data=(group, name, node, True))
                else:
                    dpg.add_checkbox(default_value=param.value, callback=param_updated, user_data=(group, name, node))
            elif isinstance(param, FloatParam):
                with dpg.group(horizontal=True) as input_group:
                    # parameter is a float
                    Window().param_input_fields.append(
                        dpg.add_input_text(
                            width=50,
                            scientific=True,
                            default_value=str(param.value),
                            callback=param_updated,
                            user_data=(group, name, node, input_group, float),
                        )
                    )
                    dpg.add_slider_float(
                        default_value=param.value,
                        min_value=param.vmin,
                        max_value=param.vmax,
                        user_data=(group, name, node, input_group, float),
                        callback=param_updated,
                    )
            elif isinstance(param, IntParam):
                with dpg.group(horizontal=True) as input_group:
                    # parameter is an integer
                    Window().param_input_fields.append(
                        dpg.add_input_text(
                            width=50,
                            default_value=str(param.value),
                            callback=param_updated,
                            user_data=(group, name, node, input_group, int),
                        )
                    )
                    dpg.add_slider_int(
                        default_value=param.value,
                        min_value=param.vmin,
                        max_value=param.vmax,
                        callback=param_updated,
                        user_data=(group, name, node, input_group, int),
                    )
            elif isinstance(param, StringParam):
                # parameter is a string
                if param.options is None:
                    # `options` is not set, use an unconstrained text input
                    Window().param_input_fields.append(
                        dpg.add_input_text(
                            default_value=param.value,
                            callback=param_updated,
                            user_data=(group, name, node),
                        )
                    )
                else:
                    # `options` is set, use a dropdown menu
                    dpg.add_combo(
                        default_value=param.value,
                        items=param.options,
                        callback=param_updated,
                        user_data=(group, name, node),
                    )
            else:
                raise NotImplementedError(f"Parameter type {type(param)} not implemented.")


def add_output_slot(parent: int, name: str, closed: bool = False, size: Tuple[int, int] = (175, 100)) -> int:
    """
    Add an output slot to the GUI, which consists of a collapsable header and a content window for the data viewer.

    ### Parameters
    `parent` : int
        The parent item (the node attribute).
    `name` : str
        The name of the output slot.
    `closed` : bool
        Whether the headers should be closed initially.
    `size` : Tuple[int, int]
        The size of the content area.

    ### Returns
    `content` : int
        The content window item.
    """

    def header_callback(_1, _2, items: List[int]):
        """
        Event callback for the collapsable header, both for general clicks, and for toggle_open events.
        This is a workaround for the fact that toggle_close events don't exist. As toggle_open is always
        called after click, clicking on the header always closes it, so the toggle_open callback is used
        reopen it if the internal header state is open.

        ### Parameters
        `items` : List[int]
            A list of four items: a bool indicating to close the header, the header, the window, and the content window.
        """
        header, window, content = items
        header_height = dpg.get_item_state(header)["rect_size"][1]
        viewer = dpg.get_item_user_data(content)[1]

        collapse = not dpg.get_item_user_data(header)
        viewer.collapsed = collapse
        viewer.set_size(propagate=False)
        dpg.set_item_user_data(header, collapse)

        if header_height == 0:
            # this happens when the header is off screen, set to default height
            header_height = 15

        if collapse:
            # header was closed, shrink window to header size
            dpg.set_item_height(window, header_height)
        else:
            # header was opened, expand window to header size plus cotent size
            if viewer is None:
                content_height = dpg.get_item_state(content)["rect_size"][1]
                if content_height == 0:
                    # this happens when the content window is off screen, set to default height
                    content_height = size[1]
            else:
                content_height = viewer.height

            dpg.set_item_height(window, content_height + header_height + 4)  # magic + 4 to avoid scroll bar

    # NOTE: we initially set the height to 1 to avoid the node reaching out of the window, which would break the
    # initial header_callback call as the header and content sizes according to DearPyGui would be 0.
    # The child window is needed, because otherwise the header will expand to fill the entire node editor window.
    with dpg.child_window(width=size[0], height=1, parent=parent) as window:
        # add collapsable content area for the data viewer
        header = dpg.add_collapsing_header(label=name, default_open=not closed, open_on_arrow=False, closable=False, user_data=closed)
        content = dpg.add_child_window(width=size[0], height=size[1], user_data=(size, None), parent=header)
        with dpg.tooltip(parent=content, show=True, delay=0.25, hide_on_activity=True):
            dpg.add_text("Ctrl + Click: switch between data viewers", bullet=True)
            dpg.add_text("Scroll: zoom in/out", bullet=True)
            dpg.add_text("Shift + Scroll: zoom horizontally", bullet=True)

    # register click handlers
    with dpg.item_handler_registry() as reg:
        dpg.add_item_clicked_handler(callback=header_callback, user_data=(header, window, content))
    dpg.bind_item_handler_registry(header, reg)
    return content


class Window:
    """
    The graphical user interface window, implemented as a threaded singleton. The window is created
    when Window() is first called. Subsequent calls to Window() will return the same instance. The
    window thread is a daemon thread, so it will be terminated when the main thread exits, or when
    Window().close() is called.

    ### Parameters
    `manager` : Manager
        The manager instance.
    """

    _instance = None
    _initialized = False

    def __new__(cls, manager=None):
        if cls._instance is None:
            # make sure the window is created in the main thread (required by MacOS)
            if threading.current_thread() != threading.main_thread():
                raise RuntimeError("Window must be created in the main thread.")

            # instantiate the window
            # TODO: add proper logging
            print("Starting graphical user interface.")
            cls._instance = super(Window, cls).__new__(cls)
            cls._instance._initialize(manager)
        return cls._instance

    @running
    @update_minimap
    def add_node(
        self,
        node_name: str,
        node: NodeRef,
        pos: Optional[Tuple[int, int]] = None,
        offset: Optional[Tuple[int, int]] = None,
        viewers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Outward facing method to add a node to the GUI.

        ### Parameters
        `node_name` : str
            The name of the node.
        `node` : NodeRef
            The node reference.
        `pos` : Optional[Tuple[int, int]]
            Position of the node. If None, uses the mouse position.
        `offset` : Optional[Tuple[int, int]]
            Offsets the node position by the given amount.
        `viewers` : Optional[List[Dict[str, Any]]]
            List of data viewer configurations.
        """
        if pos is None:
            pos = self._node_editor_mouse_pos()
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])

        # TODO: revert to using just the node_name in all cases once https://github.com/hoffstadt/DearPyGui/issues/2444 is fixed
        node_display_name = " " + node_name + " " if len(node.input_slots) == 0 and len(node.output_slots) == 0 else node_name

        with dpg.node(parent=self.node_editor, label=node_display_name, pos=pos, user_data=node) as node_id:
            ############### input slots ###############
            in_slots = {}
            for name, dtype in node.input_slots.items():
                # create input slot
                in_slots[name] = dpg.add_node_attribute(label=name, attribute_type=dpg.mvNode_Attr_Input, shape=DTYPE_SHAPE_MAP[dtype], user_data=dtype)
                # simply add a text label
                dpg.add_text(name, parent=in_slots[name])

            ############### output slots ###############
            out_slots = {}
            output_draw_handlers = {}
            for name, dtype in node.output_slots.items():
                # create output slot
                out_slots[name] = dpg.add_node_attribute(label=name, attribute_type=dpg.mvNode_Attr_Output, shape=DTYPE_SHAPE_MAP[dtype], user_data=dtype)

                # determine data viewer configuration
                viewer_kwargs = viewers[name] if viewers is not None and name in viewers else {}
                collapsed = viewer_kwargs.pop("collapsed", len(node.output_slots) > 2)

                # create content window for data viewer (initialize closed if more than two output slots)
                content = add_output_slot(out_slots[name], name, closed=collapsed)

                # create data viewer and store in content window's user data
                output_draw_handlers[name] = ViewerContainer(dtype, content, collapsed, output_draw_handlers, **viewer_kwargs)
                dpg.set_item_user_data(content, (dpg.get_item_user_data(content)[0], output_draw_handlers[name]))

            # add node to node list
            self.nodes[node_name] = GUINode(node_id, in_slots, out_slots, output_draw_handlers, node)

            # TODO: register PROCESSING_ERROR message handler and display error messages
            node.set_message_handler(MessageType.PROCESSING_ERROR, partial(self._processing_error_callback, node_name=node_name))

            # register data message handler to update the data viewers
            node.set_message_handler(MessageType.DATA, partial(handle_data, self, self.nodes[node_name]))

    @running
    def remove_node(self, name: str) -> None:
        """
        Outward facing method to remove a node from the GUI.

        ### Parameters
        `name` : str
            The name of the node.
        """
        self._remove_node(self.nodes[name], notify_manager=False)

    @running
    @update_minimap
    def _remove_node(self, item: int, notify_manager: bool = True) -> None:
        """
        Remove a node from the GUI.

        ### Parameters
        `item` : int
            The node item.
        `notify_manager` : bool
            Whether to notify the manager to remove the node.
        """
        if item in dpg.get_selected_nodes(self.node_editor):
            # deselect node if it is selected
            self._select_node(None)

        # determine the node name
        # NOTE: the strip is required for now as nodes without in- and outputs have leading and trailing spaces (https://github.com/hoffstadt/DearPyGui/issues/2444)
        name = dpg.get_item_label(item).strip()

        # remove links associated with node
        for link in list(self.links.keys()):
            if link[0] == name or link[1] == name:
                self._remove_link(self.links[link], notify_manager=False)

        # unregister node by deleting it from the node list
        del self.nodes[name]

        # delete node from gui
        dpg.delete_item(item)

        if notify_manager:
            # remove node from manager
            self.manager.remove_node(name, notify_gui=False)

    @running
    def add_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        """
        Outward facing method to add a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        """
        # determine slot IDs
        slot1 = self.nodes[node_out].output_slots[slot_out]
        slot2 = self.nodes[node_in].input_slots[slot_in]
        # add link internally
        self._add_link(self.node_editor, (slot1, slot2), notify_manager=False)

    @running
    def _add_link(self, sender: int, items: Tuple[int, int], notify_manager: bool = True) -> None:
        """
        Add a link between two nodes to the GUI.

        ### Parameters
        `sender` : int
            The sender item.
        `items` : Tuple[int, int]
            The two items to link.
        `notify_manager` : bool
            Whether to notify the manager to add the link.
        """
        # get slot names
        slot1 = dpg.get_item_label(items[0])
        slot2 = dpg.get_item_label(items[1])
        # get node names
        # NOTE: the strip is required for now as nodes without in- and outputs have leading and trailing spaces (https://github.com/hoffstadt/DearPyGui/issues/2444)
        node1 = dpg.get_item_label(dpg.get_item_parent(items[0])).strip()
        node2 = dpg.get_item_label(dpg.get_item_parent(items[1])).strip()

        # make sure the data types match
        dtype1 = dpg.get_item_user_data(items[0])
        dtype2 = dpg.get_item_user_data(items[1])
        if dtype1 != dtype2:
            # TODO: add proper logging
            print(f"Data types of slots {slot1} ({dtype1.name}) and {slot2} ({dtype2.name}) do not match.")
            return

        # remove link first if the input slot already has a link
        for link in list(self.links.keys()):
            if link[1] == node2 and link[3] == slot2:
                self._remove_link(self.links[link], notify_manager=True)

        # register link
        self.links[(node1, node2, slot1, slot2)] = dpg.add_node_link(items[0], items[1], parent=sender)

        if notify_manager:
            # add link to manager
            self.manager.add_link(node1, node2, slot1, slot2, notify_gui=False)

    def remove_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        """
        Outward facing method to remove a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        """
        self._remove_link(self.links[(node_out, node_in, slot_out, slot_in)], notify_manager=False)

    @running
    def _remove_link(self, item: int, notify_manager: bool = True) -> None:
        """
        Remove a link between two nodes from the GUI.

        ### Parameters
        `item` : int
            The link item.
        `notify_manager` : bool
            Whether to notify the manager to remove the link.
        """
        conf = dpg.get_item_configuration(item)
        # get slot names
        slot1, slot2 = dpg.get_item_label(conf["attr_1"]), dpg.get_item_label(conf["attr_2"])
        # get node names
        # NOTE: the strip is required for now as nodes without in- and outputs have leading and trailing spaces (https://github.com/hoffstadt/DearPyGui/issues/2444)
        node1 = dpg.get_item_label(dpg.get_item_parent(conf["attr_1"])).strip()
        node2 = dpg.get_item_label(dpg.get_item_parent(conf["attr_2"])).strip()

        if notify_manager:
            # remove link from manager
            self.manager.remove_link(node1, node2, slot1, slot2, notify_gui=False)

        # unregister link
        del self.links[(node1, node2, slot1, slot2)]
        # delete link from gui
        dpg.delete_item(item)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a goofi-pipe from a file.

        ### Parameters
        `path` : Optional[str]
            Optional path to load the GUI state from. If None, opens a file dialog.
        """

        def file_callback(_, info):
            self.load(path=info["file_path_name"])

        if path is None:
            # no save path set, open save dialog
            self._get_file(file_callback, message="Select a file to load from.")
            return

        try:
            # try loading the manager
            self.manager.load(filepath=path)
        except FileNotFoundError:
            # file does not exist, open error dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Error", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The file does not exist.")

                dpg.add_separator()
                dpg.add_text(path)
                dpg.add_separator()

                # add button
                dpg.add_button(label="Ok", callback=lambda _, __: dpg.delete_item(win))
        except RuntimeError:
            # file does not exist, open error dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Error", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The current pipeline is not empty.")

                # add button
                dpg.add_button(label="Ok", callback=lambda _, __: dpg.delete_item(win))

    def save(
        self,
        path: Optional[str] = None,
        overwrite: bool = True,
        save_as: bool = False,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Save the current state of the GUI.

        ### Parameters
        `path` : Optional[str]
            Optional path to save the GUI state to. If None, uses the current save path.
        `overwrite` : bool
            Whether to overwrite the current save path.
        `save_as` : bool
            Whether to open a file dialog to select a save path.
        `callback` : Optional[Callable]
            Optional callback function to call when the save is complete.
        """

        def file_callback(_, info):
            # save the manager to the selected path
            self.save(path=info["file_path_name"], overwrite=False, callback=callback)

        if save_as or (self.manager.save_path is None and path is None):
            # no save path set, open save dialog
            self._get_file(file_callback, message="Select a file to save to.")
            return

        try:
            # try saving the manager
            self.manager.save(filepath=path, overwrite=overwrite)

            if callback is not None:
                callback()
        except FileExistsError:

            def confirm_callback(_1, _2, data):
                """Callback for the overwrite confirmation dialog."""
                win, confirm = data
                if confirm:
                    self.save(path=path, overwrite=True, callback=callback)
                # close the confirmation dialog
                dpg.delete_item(win)

            # file already exists, open overwrite dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Overwrite?", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The file already exists. Overwrite?")

                if path is not None:
                    dpg.add_separator()
                    dpg.add_text(path)
                    dpg.add_separator()

                # add buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Yes", callback=confirm_callback, user_data=(win, True))
                    dpg.add_button(label="Cancel", callback=confirm_callback, user_data=(win, False))
        except RuntimeError as e:
            # something went wrong, open error dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Error", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text(str(e))

                # add button
                dpg.add_button(label="Ok", callback=lambda _, __: dpg.delete_item(win))

    def _get_file(self, callback: Callable, message: Optional[str] = None) -> None:
        """
        Open a file dialog (e.g. for loading or saving a file).

        ### Parameters
        `callback` : Callable
            The callback function to call when the file is selected.
        `message` : Optional[str]
            Optional message to display in the save dialog.
        """

        # open file selection dialog
        width, height = dpg.get_viewport_width() / 2, dpg.get_viewport_height() / 2
        with dpg.file_dialog(
            label=message,
            callback=callback,
            default_path=".",
            default_filename="",
            modal=True,
            width=width,
            height=height,
        ) as self.file_selection_window:
            dpg.add_file_extension(".gfi", label="goofi-pipes")
            dpg.add_file_extension(".*", label="All Files")

    def get_node_state(self, name: str) -> Dict[str, Any]:
        """
        Get the state of a node.

        ### Parameters
        `name` : str
            The name of the node.

        ### Returns
        `state` : Dict[str, Any]
            The state of the node.
        """
        if name not in self.nodes:
            # TODO: add proper logging
            print(f"Can't get state of node {name}. The node does not exist.")
            return None

        node = self.nodes[name]
        return {
            "pos": dpg.get_item_pos(node.item),
            "viewers": {slot: viewer.get_state() for slot, viewer in node.output_draw_handlers.items()},
        }

    def _select_node(self, item: Optional[int]) -> None:
        """
        Register or unregister a node as selected. If a new node is selected, the parameters window is
        updated to show the parameters of that node.

        ### Parameters
        `item` : Optional[int]
            The node item. If None, deselects the currently selected node.
        """
        if self.selected_node == item:
            # do nothing if the node is already selected
            return

        # update window layout
        self.selected_node = item
        self.resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()])
        # clear parameters window
        dpg.delete_item(self.side_panel_win, children_only=True)

        # NOTE: the strip is required for now as nodes without in- and outputs have leading and trailing spaces (https://github.com/hoffstadt/DearPyGui/issues/2444)
        if item is None or dpg.get_item_label(item).strip() not in self.nodes:
            self.selected_node = None
            # node deselected, hide parameters window and resize node editor
            dpg.configure_item(self.side_panel_win, show=False)
            self.resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()])
            return

        # get node reference
        # NOTE: the strip is required for now as nodes without in- and outputs have leading and trailing spaces (https://github.com/hoffstadt/DearPyGui/issues/2444)
        node = self.nodes[dpg.get_item_label(item).strip()]
        node_ref = node.node_ref

        with dpg.child_window(height=dpg.get_viewport_height() / 2, parent=self.side_panel_win):
            # add title
            dpg.add_text("Parameters")
            dpg.add_separator()

            # populate parameters window
            self.param_input_fields.clear()
            with dpg.tab_bar():
                for group in node_ref.params:
                    with dpg.tab(label=format_name(group)) as tab:
                        with dpg.table(header_row=False, parent=tab, policy=dpg.mvTable_SizingStretchProp) as table:
                            dpg.add_table_column()
                            dpg.add_table_column()

                            for name, param in node_ref.params[group].items():
                                add_param(table, group, name, param, node_ref)

                # add info tab
                with dpg.tab(label="Info"):
                    # TODO: replace magic padding number by retrieving parent width
                    dpg.add_text(node_ref.__doc__, wrap=PARAM_WINDOW_WIDTH - 70)

        with dpg.child_window(autosize_y=True, parent=self.side_panel_win):
            # add title
            dpg.add_text("Outputs")
            dpg.add_separator()

            # add one metadata view for each output slot
            self.metadata_view = {}
            with dpg.tab_bar():
                for slot in node_ref.output_slots:
                    with dpg.tab(label=slot) as tab:
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(
                                label="log-scale x-axis",
                                default_value=node.output_draw_handlers[slot].log_scale_x,
                                callback=toggle_log_plot,
                                user_data=(self, node, slot, "x"),
                            )
                            dpg.add_checkbox(
                                label="log-scale y-axis",
                                default_value=node.output_draw_handlers[slot].log_scale_y,
                                callback=toggle_log_plot,
                                user_data=(self, node, slot, "y"),
                            )
                        dpg.add_separator()
                        self.metadata_view[slot] = dpg.add_input_text(default_value="", multiline=True, readonly=True, width=-1, height=-1)

        # show parameters window
        dpg.configure_item(self.side_panel_win, show=True)

    def _processing_error_callback(self, node: NodeRef, message: Message, node_name: str) -> None:
        """Callback for the `MessageType.PROCESSING_ERROR` message type."""
        error = message.content["error"]
        if error is None:
            # no error, clear error message
            self.nodes[node_name].set_error(None, self)
            return

        self.nodes[node_name].set_error(error, self)
        print(f"Error in node {node_name}:\n{error}")

    def link_callback(self, sender: int, items: Tuple[int, int]) -> None:
        """Callback from DearPyGui that two nodes were connected."""
        self._add_link(sender, items)

    def delink_callback(self, _, item: int) -> None:
        """Callback from DearPyGui that a link was removed."""
        self._remove_link(item)

    def resize_callback(self, _, size: Tuple[int, int]) -> None:
        """Callback from DearPyGui that the viewport was resized."""
        # resize window to fill viewport
        if platform.system() == "Windows":
            # frame border seems to be different on Windows
            dpg.configure_item(self.window, width=size[0] - 16, height=size[1] - 39)
        else:
            dpg.configure_item(self.window, width=size[0], height=size[1])

        if self.selected_node is None:
            # no node selected, resize node editor to fill viewport
            dpg.configure_item(self.node_editor, width=0)
        else:
            # node selected, resize node editor to fill viewport minus parameters window
            dpg.configure_item(self.node_editor, width=size[0] - PARAM_WINDOW_WIDTH)

            children = dpg.get_item_children(self.side_panel_win)[1]
            if len(children) > 0:
                dpg.set_item_height(children[0], dpg.get_viewport_height() / 2)

    def update_title(self) -> None:
        """Update the window title."""
        start = time.time()
        while not hasattr(self, "manager"):
            # wait until the manager is initialized
            if time.time() - start > 1:
                raise RuntimeError("Timeout while waiting for manager to initialize.")
            time.sleep(0.01)

        # update window title
        unsaved_changes = "*" if self.manager.unsaved_changes else ""
        if self.manager.save_path is None:
            dpg.set_viewport_title(f"{unsaved_changes}goofi-pipe")
        else:
            dpg.set_viewport_title(f"goofi-pipe | {unsaved_changes}{self.manager.save_path.split(os.sep)[-1]}")

    def exit_callback(self, value):
        """Callback from DearPyGui that the close button was pressed."""
        if self.manager.unsaved_changes and not self.unsaved_changes_dialog_open:

            def callback(_1, _2, data):
                """Callback for the unsaved changes dialog."""
                win, save = data
                # close the confirmation dialog
                dpg.delete_item(win)
                self.unsaved_changes_dialog_open = False

                if save:
                    # save changes first
                    self.save(callback=self.terminate)
                else:
                    # close without saving
                    self.terminate()

            # unsaved changes, open confirm dialog
            self.unsaved_changes_dialog_open = True
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Save before closing?", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("There are unsaved changes.\nDo you want to save before closing?")

                # add buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=callback, user_data=(win, True))
                    dpg.add_button(label="Close without saving", callback=callback, user_data=(win, False))
            return

        # no unsaved changes, close window
        self.terminate()

    def _to_node_editor_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position from window space to node editor space."""
        # get reference node position
        orig_ref_pos = dpg.get_item_user_data("_ref")
        curr_ref_pos = dpg.get_item_state("_ref")["rect_min"]
        # calculate mouse position within node editor
        return pos[0] - curr_ref_pos[0] + orig_ref_pos[0], pos[1] - curr_ref_pos[1] + orig_ref_pos[1]

    def _to_window_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position from node editor space to window space."""
        # get reference node position
        orig_ref_pos = dpg.get_item_user_data("_ref")
        curr_ref_pos = dpg.get_item_state("_ref")["rect_min"]
        # calculate mouse position within node editor
        return pos[0] - curr_ref_pos[0] + orig_ref_pos[0], pos[1] - curr_ref_pos[1] + orig_ref_pos[1]

    def _node_editor_mouse_pos(self) -> Tuple[int, int]:
        """Get the mouse position within the node editor."""
        return self._to_node_editor_coords(dpg.get_mouse_pos(local=False))

    def _register_node_category_themes(self, darkness: float = 0.5) -> None:
        """Register themes for each node category."""
        cats = [n.category() for n in list_nodes()]
        cats = sorted(list(set(cats)))

        def scale(color, s):
            return [max(min(int(c * s), 255), 0) for c in color]

        # set up node themes
        self.node_themes = {}
        for i, cat in enumerate(cats):
            for suffix in ["", "_empty"]:
                with dpg.theme() as theme:
                    with dpg.theme_component():
                        if suffix == "_empty":
                            # NOTE: setting horizontal padding to 0 prevents a bug in dearpygui>2 where empty nodes grow indefinitely
                            # TODO: remove the extra case for empty nodes once the following issue is resolved: https://github.com/hoffstadt/DearPyGui/issues/2444
                            dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 7, category=dpg.mvThemeCat_Nodes)

                        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, scale(NODE_CAT_COLORS[i], darkness), category=dpg.mvThemeCat_Nodes)
                        dpg.add_theme_color(
                            dpg.mvNodeCol_TitleBarHovered,
                            scale(NODE_CAT_COLORS[i], darkness + 0.1),
                            category=dpg.mvThemeCat_Nodes,
                        )
                        dpg.add_theme_color(
                            dpg.mvNodeCol_TitleBarSelected,
                            scale(NODE_CAT_COLORS[i], darkness + 0.2),
                            category=dpg.mvThemeCat_Nodes,
                        )

                self.node_themes[cat + suffix] = theme

        # set up node error themes
        self.node_error_themes = {}
        for i, cat in enumerate(cats):
            with dpg.theme() as theme:
                with dpg.theme_component():
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [153, 20, 22], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [173, 20, 22], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [193, 20, 22], category=dpg.mvThemeCat_Nodes)

                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, scale(NODE_CAT_COLORS[i], darkness), category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, scale(NODE_CAT_COLORS[i], darkness + 0.1), category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(
                        dpg.mvNodeCol_TitleBarSelected,
                        scale(NODE_CAT_COLORS[i], darkness + 0.2),
                        category=dpg.mvThemeCat_Nodes,
                    )
            self.node_error_themes[cat] = theme

    def _initialize(self, manager, width=1280, height=720):
        """Initialize the window and launch the event loop (blocking)."""
        dpg.create_context()
        self.manager = manager

        # initialize dicts to map names to dpg items
        self.nodes = {}
        self.links = {}
        self.selected_node = None
        self.create_node_window = None
        self.last_create_node_tab = 0
        self.file_selection_window = None
        self.unsaved_changes_dialog_open = False
        self.node_clipboard = None
        self.node_info_window = None
        self.metadata_view = None
        self.param_input_fields = []

        # create window
        self.window = dpg.add_window(
            label="window",
            width=width,
            height=height,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
        )

        # create menu bar
        with dpg.menu_bar(parent=self.window):
            dpg.add_menu_item(label="Load", callback=lambda: self.load())
            dpg.add_menu_item(label="Save As", callback=lambda: self.save(save_as=True))
            dpg.add_menu_item(label="Save", callback=lambda: self.save())

        with dpg.group(horizontal=True, parent=self.window):
            # create node editor
            self.node_editor = dpg.add_node_editor(callback=self.link_callback, delink_callback=self.delink_callback)
            # add parameters window
            self.side_panel_win = dpg.add_child_window(label="Parameters", autosize_x=True, border=False)

        self._register_node_category_themes()

        # hide the reference node using themes
        with dpg.theme() as ref_theme:
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 0, category=dpg.mvThemeCat_Nodes)

        # create a reference node to calculate the mouse position within the node editor
        # NOTE: this is a workaround as DearPyGui doesn't provide access to the node editor coordinates
        # NOTE: ideally we would set show=False in the ref node, but this causes a Segmentation Fault
        # NOTE: show=False is now fixed (https://github.com/hoffstadt/DearPyGui/pull/2225), waiting for release
        pos = [0, 0]
        dpg.add_node(label=" ", tag="_ref", parent=self.node_editor, pos=pos, user_data=pos, draggable=False)
        dpg.bind_item_theme("_ref", ref_theme)

        # register user interaction handlers
        with dpg.handler_registry():
            dpg.add_key_release_handler(callback=events.key_release_callback, user_data=self)
            dpg.add_mouse_click_handler(callback=events.click_callback, user_data=self)
            dpg.add_mouse_double_click_handler(callback=events.double_click_callback, user_data=self)

        # register viewport resize handler
        dpg.set_viewport_resize_callback(self.resize_callback)

        # register exit handler
        dpg.set_exit_callback(self.exit_callback)

        # set up window
        self.resize_callback(None, [width, height])

        # start DearPyGui
        dpg.create_viewport(title="goofi-pipe", width=width, height=height, disable_close=True)
        assets_path = pkg_resources.files(assets)
        dpg.set_viewport_large_icon(assets_path / "goofi.ico")
        dpg.set_viewport_small_icon(assets_path / "goofi.ico")
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # we are done initializing, start the event loop (blocking)
        self._initialized = True
        dpg.start_dearpygui()

        # cleanup after event loop
        dpg.destroy_context()

        # TODO: add proper logging
        print("Shutting down graphical user interface.")

        # terminate manager
        manager.terminate(notify_gui=False)

    def terminate(self):
        """Stop DearPyGui and close the window."""
        dpg.stop_dearpygui()
