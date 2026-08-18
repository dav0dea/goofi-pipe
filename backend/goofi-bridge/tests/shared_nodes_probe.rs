//! Proves the shared test-node library is reachable from another crate's integration test — the
//! property that lets one headless test stand in for a unit test. Delete this file only together
//! with the library it checks.
use goofi_engine::Graph;

#[test]
fn an_integration_test_in_another_crate_can_add_a_shared_test_node() {
    let mut g = Graph::new();
    for ty in ["_TestEcho", "_TestFail", "_TestCounter", "_TestRequired", "_TestPicker"] {
        g.add_node(ty, None).unwrap_or_else(|e| panic!("{ty} unreachable from goofi-bridge: {e}"));
    }
    assert_eq!(g.node_count(), 5);
}
