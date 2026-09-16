//! Real-WSL2 end-to-end integration, compiled as a **lib** test module
//! (`#[cfg(test)]`) so it runs under the known-good lib test harness
//! (`cargo test --lib`). The separate `tests/` integration-test harness
//! binary on this machine fails to LOAD (0xc0000139 / 0xc0000135, a broken
//! link of the extra test-runner deps), so all WSL-gated tests live here.
//!
//! Run (needs internet + a CUDA GPU visible from inside WSL2; provisions an
//! isolated venv under ~/llm-lp). First provision downloads vLLM + torch
//! (~several GB), so allow 20–60 min. Set LLM_TEST_WSL=1 to actually run:
//!
//!   LLM_TEST_WSL=1 cargo test --lib -- --ignored --nocapture wsl_it
//!
//! The test is `#[ignore]`d by default and is a no-op unless LLM_TEST_WSL is set.

use crate::provision;
use crate::server;
use crate::state::{AppState, ServerDef};
use crate::wsl;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Must match the WSL distro the user picks in Settings (here: the default).
const DISTRO: &str = "Ubuntu-22.04";
const VENV_DIR: &str = "~/llm-lp/.venv";

fn http() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("http client")
}

fn wait_health(http: &reqwest::blocking::Client, port: u16, timeout_s: u64) {
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + Duration::from_secs(timeout_s);
    let mut last = String::new();
    while Instant::now() < deadline {
        match http.get(&url).send() {
            Ok(r) if r.status().is_success() => {
                println!("{url} healthy");
                return;
            }
            Ok(r) => last = format!("status {}", r.status()),
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_secs(3));
    }
    panic!("{url} not healthy within {timeout_s}s (last: {last})");
}

#[test]
#[ignore]
fn wsl_it() {
    if std::env::var("LLM_TEST_WSL").is_err() {
        eprintln!("skipped: LLM_TEST_WSL not set");
        return;
    }

    // --- 1. Provision twice (idempotent) ---
    let t0 = Instant::now();
    let rep1 = provision::provision_all(DISTRO, VENV_DIR, |p, l| println!("  [{p}] {l}"))
        .expect("provision #1");
    println!(
        "provision #1 ok in {:?}: vllm={:?} torch={:?} cuda={} gpu={:?} bf16={}",
        t0.elapsed(),
        rep1.vllm_version,
        rep1.torch_version,
        rep1.cuda_available,
        rep1.gpu_name,
        rep1.bf16_supported
    );
    assert!(rep1.cuda_available, "torch must see CUDA inside WSL");
    assert!(rep1.vllm_version.is_some(), "vllm must be importable");

    let t1 = Instant::now();
    let _rep2 = provision::provision_all(DISTRO, VENV_DIR, |_p, _l| {})
        .expect("provision #2 (idempotent)");
    println!("provision #2 ok in {:?} (idempotent path)", t1.elapsed());

    // --- 2. Two servers: instruct (0.5B) + embed (bge-small) ---
    let state = Arc::new(AppState::new());
    {
        let mut cfg = state.config.lock().unwrap();
        cfg.distro = DISTRO.to_string();
        cfg.venv_dir = VENV_DIR.to_string();
        cfg.servers.clear();
        cfg.servers.push(ServerDef {
            id: "it-qwen".into(),
            name: "qwen-0.5b".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            task: "instruct".into(),
            port: 8130,
            gpu_mem_util: 0.55,
            max_model_len: Some(2048),
            quant: "fp16".into(),
            served_model_name: Some("qwen-0.5b".into()),
            params_b: Some(0.494),
        });
        cfg.servers.push(ServerDef {
            id: "it-bge".into(),
            name: "bge-small".into(),
            model_id: "BAAI/bge-small-en-v1.5".into(),
            task: "embed".into(),
            port: 8131,
            gpu_mem_util: 0.15,
            max_model_len: Some(512),
            quant: "fp16".into(),
            served_model_name: Some("embedder".into()),
            params_b: Some(0.033),
        });
        cfg.save().expect("save config");
    }

    // --- 3. Start both; poll /health ---
    println!("starting servers…");
    server::start_server(&state, None, "it-qwen").expect("start instruct");
    server::start_server(&state, None, "it-bge").expect("start embed");
    let h = http();
    wait_health(&h, 8130, 480);
    wait_health(&h, 8131, 480);

    // --- 4. /v1/models lists both served names ---
    let models1: serde_json::Value = h
        .get("http://127.0.0.1:8130/v1/models")
        .send()
        .expect("GET /v1/models #1")
        .json()
        .expect("models json");
    let models2: serde_json::Value = h
        .get("http://127.0.0.1:8131/v1/models")
        .send()
        .expect("GET /v1/models #2")
        .json()
        .expect("models json");
    let names1: Vec<String> = models1["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect();
    let names2: Vec<String> = models2["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect();
    println!("instruct models: {names1:?}\nembed models: {names2:?}");
    assert!(
        names1.iter().any(|n| n.contains("qwen-0.5b")),
        "qwen served name missing: {names1:?}"
    );
    assert!(
        names2.iter().any(|n| n.contains("embedder")),
        "embedder served name missing: {names2:?}"
    );

    // --- 5. Chat completion through the playground path ---
    let body = serde_json::json!({
        "model": "qwen-0.5b",
        "messages": [{"role": "user", "content": "Reply with the exact words: integration ok"}],
        "max_tokens": 32
    });
    let chat: serde_json::Value = h
        .post("http://127.0.0.1:8130/v1/chat/completions")
        .json(&body)
        .send()
        .expect("chat")
        .json()
        .expect("chat json");
    let content = chat["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    println!("chat: {content:?}");
    assert!(!content.trim().is_empty(), "chat must return a completion");

    // --- 6. /metrics counters moved ---
    let metrics = h
        .get("http://127.0.0.1:8130/metrics")
        .send()
        .expect("metrics")
        .text()
        .expect("metrics text");
    let m = server::parse_metrics(&metrics).expect("parse metrics");
    println!(
        "metrics: rtoks={} ptoks={} running={}",
        m.total_generation_tokens, m.total_prompt_tokens, m.running
    );
    assert!(
        m.total_generation_tokens > 0,
        "generation counter must move after a chat"
    );

    // --- 7. Embedding endpoint ---
    let emb: serde_json::Value = h
        .post("http://127.0.0.1:8131/v1/embeddings")
        .json(&serde_json::json!({ "model": "embedder", "input": "hello world" }))
        .send()
        .expect("embeddings")
        .json()
        .expect("embeddings json");
    let dim = emb["data"][0]["embedding"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    println!("embedding dim = {dim}");
    assert!(dim > 0, "embedding must have a vector");

    // --- 8. Stop both via server::stop_server (SIGTERM→taskkill fallback) ---
    for id in ["it-qwen", "it-bge"] {
        let pidline = wsl::run_script(DISTRO, &format!("cat ~/llm-lp/run/{id}.pid 2>/dev/null || true"))
            .stdout;
        let pid: Option<u32> = pidline.trim().parse().ok();
        println!("{id} pidfile: '{pidline:?}'");
        server::stop_server(&state, None, id).expect("stop");
        if let Some(pid) = pid {
            std::thread::sleep(Duration::from_millis(1500));
            let alive = wsl::run_script(DISTRO, &format!("kill -0 {pid} 2>/dev/null && echo alive || echo gone"));
            println!("{id} pid {pid}: {}", alive.stdout.trim());
            assert_eq!(alive.stdout.trim(), "gone", "{id} pid still running after stop");
        }
    }

    println!("WSL INTEGRATION OK ✓");
}