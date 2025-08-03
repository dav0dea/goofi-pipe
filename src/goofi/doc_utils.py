import sys
from os import path

import libcst as cst
from joblib import Parallel, delayed
from litellm import completion
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


class DocstringAdder(cst.CSTTransformer):
    def __init__(self, target_class, docstring):
        self.target_class = target_class
        self.docstring = docstring

    def leave_ClassDef(self, original_node, updated_node):
        if original_node.name.value == self.target_class:
            return updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=[cst.SimpleStatementLine([cst.Expr(cst.SimpleString(f'"""{self.docstring}"""'))])]
                    + [
                        s
                        for s in updated_node.body.body
                        if not (isinstance(s, cst.SimpleStatementLine) and isinstance(s.body[0], cst.Expr) and isinstance(s.body[0].value, cst.SimpleString))
                    ]
                )
            )
        return updated_node


def add_docstring_to_class(filepath, classname, docstring):
    with open(filepath, "r", encoding="utf-8") as f:
        module = cst.parse_module(f.read())
    mod_new = module.visit(DocstringAdder(classname, docstring))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(mod_new.code)


DOC_PROMPT = """
```
{}
```

The above code is the definition and implementation of a node in goofi-pipe, a node-based visual coding software for real-time signal processing.
Examine this code closely while paying special attention to the function of this node.
Your task is to generate documentation for this node with a focus on what the node is doing.
Only describe the general node's function and the input and output data.
Do NOT include the parameters in the documentation unless required to explain the function, input or output. Parameter documentation exists separately.
Write the documentation in plain text, not markdown. The documentation should be concise and accurate, do not make mistakes!
Respond with ONLY the documentation and nothing else.
Use the following structure:
```
<general description of the node>

Inputs:
- <in1_name>: <description>
- ...

Outputs:
- <out1_name>: <description>
- ...
```

Inputs are defined in `config_input_slots`, and outputs in `config_output_slots`. If a node doesn't have inputs or outputs it may not define these functions.
In that case do not include "Inputs" or "Outputs" in the documentation respectively.
"""


def _gen_doc(node):
    if node.__doc__:
        return

    clsname = node.__name__
    clspath = sys.modules[node.__module__].__file__

    with open(clspath, "r") as f:
        file_txt = f.read()

    # generate documentation for the current node
    response = completion(model="openai/gpt-4.1", messages=[{"role": "user", "content": DOC_PROMPT.format(file_txt)}])
    docstring = response.choices[0].message.content

    # add leading and trailing new lines
    docstring = f"\n{docstring.strip()}\n"

    add_docstring_to_class(clspath, clsname, docstring)


def gen_node_docs():
    """
    This function retrieves a list of node classes using `list_nodes`, then generates documentation for each node
    in parallel and updates the docstring of the node.
    """
    nodes_cls = list_nodes(verbose=True)
    Parallel(n_jobs=-1)(delayed(_gen_doc)(node) for node in tqdm(nodes_cls, desc="Generating documentation"))


if __name__ == "__main__":
    gen_node_docs()
