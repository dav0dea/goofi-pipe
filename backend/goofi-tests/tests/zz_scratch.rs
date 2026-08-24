use goofi_tests::{hex, j, Goofi};
use serde_json::Value;
use std::time::{Duration, Instant};

fn drain(rx: &mut tokio::sync::broadcast::Receiver<String>, window: Duration) -> Vec<Value> {
    let deadline = Instant::now() + window;
    let mut out = Vec::new();
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(raw) => out.push(serde_json::from_str(&raw).unwrap()),
            Err(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    out
}

#[test]
fn scratch_stage_after_load() {
    let g = Goofi::new();
    let a = g.add("Oscillator");
    g.ready(a);
    let mut rx = g.state.events.subscribe();
    let pre = drain(&mut rx, Duration::from_millis(1500));
    let stages: Vec<&Value> = pre.iter().filter(|e| e["event"] == "node_stage").collect();
    eprintln!("PRE-LOAD stage events: {stages:?}");

    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    let mut rx = g.state.events.subscribe();
    g.call("load", j!({ "content": yaml }));
    g.ready(a);
    eprintln!("ENGINE stage after load: {} (uid {})", g.stage(a), hex(a));
    let post = drain(&mut rx, Duration::from_millis(3000));
    for e in &post {
        if e["event"] == "graph_replaced" {
            eprintln!("SNAPSHOT runtime: {}", e["payload"]["runtime"]);
        }
    }
    let stages: Vec<&Value> = post.iter().filter(|e| e["event"] == "node_stage").collect();
    eprintln!("POST-LOAD stage events: {stages:?}");
}
