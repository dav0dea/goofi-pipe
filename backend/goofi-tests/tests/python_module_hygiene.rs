//! Module hygiene, in its own process so `sys.modules` is not polluted by other tests.
//! `from_source` must pop the unique module it minted; the instance keeps it alive via `__globals__`.
#![cfg(feature = "embed")]

use goofi_python::inproc::PyNode;
use pyo3::prelude::*;
use pyo3::types::PyModule;

const NODE: &str = concat!(
    "import goofi\n",
    "import numpy as np\n",
    "class Double(goofi.Node):\n",
    "    INPUTS = {'data': goofi.DataType.ARRAY}\n",
    "    OUTPUTS = {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, data):\n",
    "        return {'out': data.data * 2.0}\n",
);

fn lingering_user_modules() -> usize {
    Python::attach(|py| {
        let modules = PyModule::import(py, "sys").unwrap().getattr("modules").unwrap();
        let keys = modules.call_method0("keys").unwrap();
        keys.try_iter()
            .unwrap()
            .filter_map(|k| k.ok()?.extract::<String>().ok())
            .filter(|k| k.starts_with("goofi_user_"))
            .count()
    })
}

#[test]
fn building_nodes_does_not_leak_modules_into_sys_modules() {
    // Keep the nodes ALIVE: the point is that the `sys.modules` entries are evicted regardless.
    let _nodes: Vec<PyNode> = (0..8)
        .map(|_| PyNode::from_source(NODE, vec![("data", false)], vec!["out"]).expect("build node"))
        .collect();
    assert_eq!(
        lingering_user_modules(),
        0,
        "PyNode module names must be evicted from sys.modules (no unbounded growth)"
    );
}
