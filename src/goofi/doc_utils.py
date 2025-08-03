from os import path

from tqdm import tqdm

from goofi.node_helpers import list_nodes

CATEGORY_DESCRIPTIONS = {
    "inputs": "Nodes that provide data to the pipeline.",
    "outputs": "Nodes that send data to external systems.",
    "analysis": "Nodes that perform analysis on the data.",
    "array": "Nodes implementing array operations.",
    "signal": "Nodes implementing signal processing operations.",
    "misc": "Miscellaneous nodes that do not fit into other categories.",
}


def update_docs():
    """
    Updates the documentation by updating the list of nodes in the README.
    """

    nodes_cls = list_nodes(verbose=True)

    nodes = dict()
    for node in tqdm(nodes_cls, desc="Collecting node information"):
        if node.category() not in nodes:
            nodes[node.category()] = []

        # collect the node information
        nodes[node.category()].append(
            {
                "name": node.__name__,
                "input_slots": node.config_input_slots(),
                "output_slots": node.config_output_slots(),
            }
        )

    # find the README file
    print("Loading README file...", end="")
    readme_path = path.join(path.dirname(__file__), "..", "..", "README.md")
    readme_path = path.abspath(readme_path)
    assert path.exists(readme_path), f"README file not found: {readme_path}"

    # read the README file
    with open(readme_path, "r") as f:
        readme = f.read()
    print("done")

    # find the start and end of the node list
    start_tag = "<!-- !!GOOFI_PIPE_NODE_LIST_START!! -->"
    end_tag = "<!-- !!GOOFI_PIPE_NODE_LIST_END!! -->"
    start = readme.find(start_tag)
    end = readme.find(end_tag)

    # generate the new node list
    new_nodes = []
    for category, nodes_list in tqdm(nodes.items(), desc="Generating new node list"):
        new_nodes.append(f"## {category.capitalize()}\n")
        new_nodes.append(f"{CATEGORY_DESCRIPTIONS[category]}\n")
        new_nodes.append("<details><summary>View Nodes</summary>\n")
        for node in nodes_list:
            new_nodes.append(f"<details><summary>&emsp;{node['name']}</summary>\n")
            new_nodes.append("  - **Inputs:**")
            for slot, slot_type in node["input_slots"].items():
                new_nodes.append(f"    - {slot}: {slot_type}")
            new_nodes.append("  - **Outputs:**")
            for slot, slot_type in node["output_slots"].items():
                new_nodes.append(f"    - {slot}: {slot_type}")
            new_nodes.append("  </details>\n")
        new_nodes.append("</details>\n")

    # insert the new node list into the README
    print("Updating README file...", end="")
    new_readme = readme[: start + len(start_tag)] + "\n" + "\n".join(new_nodes) + readme[end:]

    # write the updated README
    with open(readme_path, "w") as f:
        f.write(new_readme)
    print("done")
