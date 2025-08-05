import time
from copy import deepcopy

import dearpygui.dearpygui as dpg

from goofi.node_helpers import list_nodes


def is_ctrl_down() -> bool:
    generic_ctrl = False
    if hasattr(dpg, "mvKey_Control"):
        generic_ctrl = dpg.is_key_down(dpg.mvKey_Control)
    return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl) or generic_ctrl


def is_shift_down() -> bool:
    generic_shift = False
    if hasattr(dpg, "mvKey_Shift"):
        generic_shift = dpg.mvKey_Shift
    return dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift) or generic_shift


def is_click_inside(item):
    """Check if the mouse click was inside the given item."""
    mouse_pos = dpg.get_mouse_pos(local=False)
    try:
        item_pos = dpg.get_item_state(item)["rect_min"]
    except KeyError:
        item_pos = dpg.get_item_pos(item)

    item_size = dpg.get_item_rect_size(item)
    return (
        mouse_pos[0] >= item_pos[0]
        and mouse_pos[1] >= item_pos[1]
        and mouse_pos[0] <= item_pos[0] + item_size[0]
        and mouse_pos[1] <= item_pos[1] + item_size[1]
    )


################################
######### Mouse Events #########
################################


def click_callback(_, btn, win):
    """Update node selection after a click event."""
    selected = dpg.get_selected_nodes(win.node_editor)
    if len(selected) == 1:
        win._select_node(selected[0])
    else:
        win._select_node(None)

    if win.create_node_window is not None and not is_click_inside(win.create_node_window):
        # the create node window is open but the click was outside of it, close it
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None
    if win.node_info_window is not None and not is_click_inside(win.node_info_window):
        # the node info window is open but the click was outside of it, close it
        dpg.delete_item(win.node_info_window)
        win.node_info_window = None
    # TODO: do the same thing for the file selection window, which has a broken get_item_pos

    if btn == 1:
        for node in win.nodes.values():
            if is_click_inside(node.item):
                # right click on a node, open an information window
                node.display_info(win)
                break


def double_click_callback(_, btn, win):
    """If double click was left mouse button, open the window for creating a node."""
    if btn == 0 and len(dpg.get_selected_nodes(win.node_editor)) == 0:
        create_node(win)


################################
########## Key Events ##########
################################


def key_release_callback(_, data, win):
    """Handle key press events by calling the appropriate function from the key handler map."""
    if data in KEY_HANDLER_MAP:
        KEY_HANDLER_MAP[data](win)


#############################
########## Actions ##########
#############################


def delete_selected_item(win):
    """Deletes the selected node or link."""
    if any([dpg.is_item_active(item) for item in win.param_input_fields if dpg.does_item_exist(item)]):
        # an input field is active, do not delete the selected node
        return

    for node in dpg.get_selected_nodes(win.node_editor):
        win._remove_node(node)
    for link in dpg.get_selected_links(win.node_editor):
        win._remove_link(link)


def select_node_callback(_1, _2, user_data):
    """Callback for when a node is selected in the create node window."""
    win, node = user_data
    # clean up the state of the GUI
    escape(win)
    dpg.clear_selected_nodes(win.node_editor)
    dpg.clear_selected_links(win.node_editor)
    # create the node inside the manager, which will notify the window
    win.manager.add_node(node.__name__, node.category())


def create_selected_node(win):
    """Callback for when a node is selected in the create node window."""
    if win.create_node_window is None:
        return

    tab_bar, search_group, searchbox = dpg.get_item_user_data(win.create_node_window)
    if dpg.get_item_configuration(tab_bar)["show"]:
        # the tab bar is visible, the user has not selected a node yet
        return

    # get the selected node
    try:
        top_btn = dpg.get_item_children(search_group)[1][0]
    except IndexError:
        # no node was selected
        dpg.focus_item(searchbox)
        return

    # "click" the button
    user_data = dpg.get_item_user_data(top_btn)
    dpg.get_item_configuration(top_btn)["callback"](top_btn, None, user_data)


def create_node(win):
    """Opens the window for creating a node. If the window is already open, switch to the next tab."""
    # create a dictionary of nodes by category
    categories = {}
    for node in list_nodes():
        cat = node.category()
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node)

    if win.create_node_window is not None:
        if dpg.is_item_focused(win.create_node_window):
            # the window is open and focused, switch to the next tab
            tab_bar = dpg.get_item_user_data(win.create_node_window)[0]
            # increment tab index
            if is_shift_down():
                win.last_create_node_tab = (win.last_create_node_tab - 1) % len(categories)
            else:
                win.last_create_node_tab = (win.last_create_node_tab + 1) % len(categories)
            # switch to the next tab
            tab = f"tab_{list(categories.keys())[win.last_create_node_tab]}"
            dpg.set_value(tab_bar, tab)
            return

        # the window is open but not focused, close it to reopen later in this function
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None

    if not dpg.is_item_hovered(win.node_editor):
        # the node editor is not hovered, do not open the window
        return

    # create a new window instance
    win.create_node_window = dpg.add_window(label="Create Node", pos=dpg.get_mouse_pos(local=False), no_collapse=True, autosize=True)

    def search_callback(_, data):
        """Callback for when the search bar changes."""
        tab_bar, search_group, _ = dpg.get_item_user_data(win.create_node_window)

        if len(data) == 0:
            # no search query, show all nodes
            dpg.configure_item(tab_bar, show=True)
            dpg.configure_item(search_group, show=False)
            return

        # hide all tabs
        dpg.configure_item(tab_bar, show=False)
        # clear the search group
        dpg.delete_item(search_group, children_only=True)

        # add a button for each node that matches the search query
        for node in list_nodes():
            if data.lower() in node.__name__.lower():
                btn = dpg.add_button(label=node.__name__, callback=select_node_callback, user_data=(win, node), parent=search_group)
                with dpg.tooltip(btn):
                    dpg.add_text(node.docstring().strip(), wrap=500)

        # show the search group
        dpg.configure_item(search_group, show=True)

    # add search bar
    searchbox = dpg.add_input_text(hint="search", parent=win.create_node_window, callback=search_callback)
    win.param_input_fields.append(searchbox)
    dpg.add_separator(parent=win.create_node_window)

    # create a tab bar with a tab for each category
    with dpg.tab_bar(parent=win.create_node_window) as tab_bar:
        for cat, nodes in categories.items():
            with dpg.tab(label=cat, tag=f"tab_{cat}"):
                # create a button for each node in the category
                for node in nodes:
                    btn = dpg.add_button(label=node.__name__, callback=select_node_callback, user_data=(win, node))
                    with dpg.tooltip(btn):
                        dpg.add_text(node.docstring().strip(), wrap=500)

    # create a vertical group for listing nodes during search
    search_group = dpg.add_group(horizontal=False, parent=win.create_node_window, show=False)

    # switch to the current tab
    dpg.set_value(tab_bar, f"tab_{list(categories.keys())[win.last_create_node_tab]}")
    # store the tab bar and search group in the window user data
    dpg.set_item_user_data(win.create_node_window, (tab_bar, search_group, searchbox))

    # focus the search bar
    dpg.focus_item(searchbox)


def escape(win):
    """
    Provides a way to cancel the creation of a node by closing the node selection window.
    If it was already closed, clears the current node and link selection.
    """
    if win.create_node_window is not None:
        # close the create node window
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None
    elif win.file_selection_window is not None:
        # close the file selection window
        dpg.delete_item(win.file_selection_window)
        win.file_selection_window = None
    else:
        # clear the node and link selection
        dpg.clear_selected_nodes(win.node_editor)
        dpg.clear_selected_links(win.node_editor)

    if win.node_info_window is not None:
        # close the node info window
        dpg.delete_item(win.node_info_window)
        win.node_info_window = None


def save_manager(win):
    """Save the manager to a file. We go through the Window to potentially open a file selection dialog."""
    if is_ctrl_down():
        win.save()


def copy_selected_nodes(win, timeout: float = 0.1):
    """Copy the selected nodes to the clipboard."""
    if not is_ctrl_down():
        return

    # retrieve selected nodes and their positions
    nodes = {n: dpg.get_item_user_data(n) for n in dpg.get_selected_nodes(win.node_editor)}
    if len(nodes) == 0:
        return

    positions = [dpg.get_item_pos(n) for n in dpg.get_selected_nodes(win.node_editor)]
    avg_pos = [sum(p[0] for p in positions) / len(positions), sum(p[1] for p in positions) / len(positions)]

    # wait for all nodes to respond, if their serialization_pending flag is set
    start = time.time()
    serialized_nodes = []
    for (item, node), pos in zip(nodes.items(), positions):
        while node.serialization_pending and time.time() - start < timeout:
            # wait for the node to respond or for the timeout to be reached
            time.sleep(0.01)

        # get node name
        names = [name for name, gui_node in win.nodes.items() if gui_node.item == item]
        assert len(names) == 1, f"The following nodes seem to be duplicates: {names}"
        name = names[0]

        # check if we got a response in time
        if node.serialization_pending:
            # TODO: add proper logging
            print(f"WARNING: Node {name} timed out while waiting for serialization. Node state is possibly outdated.")

        if node.serialized_state is None:
            # TODO: add proper logging
            print(f"ERROR: Node {name} does not have a serialized state. Copying is not possible.")
            win.node_clipboard = None
            return

        # get links connected to input slots of the current node
        input_links = []
        for n1, n2, s1, s2 in win.links.keys():
            if n2 == name:
                input_links.append((n1, n2, s1, s2))

        # store the serialized state
        ser = deepcopy(node.serialized_state)
        ser["gui_kwargs"] = {"offset": [pos[0] - avg_pos[0], pos[1] - avg_pos[1]]}
        ser["name"] = name
        ser["input_links"] = input_links
        # we don't need the output connections
        del ser["out_conns"]
        serialized_nodes.append(ser)

    # store the serialized nodes in the clipboard
    win.node_clipboard = serialized_nodes


def paste_nodes(win):
    """Paste the nodes from the clipboard."""
    if not is_ctrl_down() or win.node_clipboard is None:
        return

    # add the nodes to the manager
    rename_nodes = {}
    for node in win.node_clipboard:
        new_name = win.manager.add_node(node["_type"], node["category"], params=node["params"], **node["gui_kwargs"])
        rename_nodes[node["name"]] = new_name

    # add links between the nodes
    for node in win.node_clipboard:
        for n1, n2, s1, s2 in node["input_links"]:
            n1 = rename_nodes[n1] if n1 in rename_nodes else n1
            n2 = rename_nodes[n2] if n2 in rename_nodes else n2
            win.manager.add_link(n1, n2, s1, s2)


# the key handler map maps key press events to functions that handle them
KEY_HANDLER_MAP = {
    dpg.mvKey_Delete: delete_selected_item,
    dpg.mvKey_Back: delete_selected_item,
    dpg.mvKey_Tab: create_node,
    dpg.mvKey_Escape: escape,
    dpg.mvKey_S: save_manager,
    dpg.mvKey_C: copy_selected_nodes,
    dpg.mvKey_V: paste_nodes,
    dpg.mvKey_Return: create_selected_node,
}
