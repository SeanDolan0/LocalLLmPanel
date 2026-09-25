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
//! When enabled, it writes server state only to a temporary test config directory.

use crate::provision;
use crate::server;
use crate::state::{AppState, ServerDef};
use crate::wsl;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
#[ignore]
fn llamacpp_it() {
    if std::env::var("LLM_TEST_LLAMACPP").is_err() {
        eprintln!("skipped: LLM_TEST_LLAMACPP not set");
        return;
    }
    let exe = std::env::var("LLM_TEST_LLAMACPP_EXE")
        .expect("LLM_TEST_LLAMACPP_EXE must point to llama-server.exe");
    let model = std::env::var("LLM_TEST_LLAMACPP_MODEL")
        .expect("LLM_TEST_LLAMACPP_MODEL must point to a tiny GGUF");
    let port = 8199u16;
    let def = ServerDef {
        backend: "llamacpp".into(),
        id: "it-llamacpp".into(),
        name: "llama.cpp integration".into(),
        model_id: model.clone(),
        task: "instruct".into(),
        port,
        gpu_mem_util: 0.8,
        max_model_len: None,
        quant: "GGUF".into(),
        served_model_name: None,
        kv_cache_dtype: None,
        llamacpp_channel: crate::state::LlamaCppChannel::Upstream,
        enforce_eager: false,
        params_b: None,
        swap_space_gb: None,
        cpu_offload_gb: None,
        was_running: false,
        model_path: Some(model.clone()),
        mmproj_path: None,
        ctx_size: Some(512),
        n_gpu_layers: Some(99),
        n_cpu_moe: None,
        fit: true,
        fit_target: None,
        device: None,
        api_key: None,
        log_verbosity: None,
        flash_attn: true,
        cache_type_k: "q8_0".into(),
        cache_type_v: "q8_0".into(),
        threads: Some(4),
        batch_size: Some(128),
        ubatch_size: Some(64),
        parallel: 1,
        jinja: true,
        no_kv_offload: false,
        metrics: true,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
    };
    let mut child = crate::wsl::NativeChild::spawn(
        std::path::Path::new(&exe),
        &server::build_llamacpp_args_with_help(&def, ""),
        |_| {},
    )
    .expect("spawn llama-server");
    let client = http();
    wait_health(&client, port, 300);
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Say hello briefly."}],
            "max_tokens": 8
        }))
        .send()
        .expect("chat request")
        .error_for_status()
        .expect("chat response");
    assert!(response.json::<serde_json::Value>().is_ok());
    child.kill().expect("kill llama-server");
    child.join();
    assert!(child.try_wait().expect("wait llama-server").is_some());
}

/// Must match the WSL distro the user picks in Settings (here: the default).
const DISTRO: &str = "Ubuntu-22.04";
const VENV_DIR: &str = "~/llm-lp/.venv";

/// Keep the opt-in integration test away from the user's real config file.
/// `PersistedConfig::path` honors this variable only in test builds.
struct TestConfigDir {
    path: std::path::PathBuf,
    previous: Option<String>,
}

impl TestConfigDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "local-llm-panel-it-config-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create isolated test config directory");
        let previous = std::env::var("LLM_TEST_CONFIG_DIR").ok();
        std::env::set_var("LLM_TEST_CONFIG_DIR", &path);
        Self { path, previous }
    }
}

impl Drop for TestConfigDir {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var("LLM_TEST_CONFIG_DIR", previous);
        } else {
            std::env::remove_var("LLM_TEST_CONFIG_DIR");
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

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
    let _rep2 =
        provision::provision_all(DISTRO, VENV_DIR, |_p, _l| {}).expect("provision #2 (idempotent)");
    println!("provision #2 ok in {:?} (idempotent path)", t1.elapsed());

    // --- 2. Two servers: instruct (0.5B) + embed (bge-small) ---
    // All config writes from server start/stop go to this isolated directory;
    // the user's real panel configuration is never overwritten.
    let _config_guard = TestConfigDir::new();
    let state = Arc::new(AppState::new());
    {
        let mut cfg = state.config.lock().unwrap();
        cfg.distro = DISTRO.to_string();
        cfg.venv_dir = VENV_DIR.to_string();
        cfg.servers.clear();
        cfg.servers.push(ServerDef {
            backend: "vllm".into(),
            id: "it-qwen".into(),
            name: "qwen-0.5b".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            task: "instruct".into(),
            port: 8130,
            gpu_mem_util: 0.55,
            max_model_len: Some(2048),
            quant: "fp16".into(),
            served_model_name: Some("qwen-0.5b".into()),
            kv_cache_dtype: None,
            llamacpp_channel: crate::state::LlamaCppChannel::Upstream,
            enforce_eager: true,
            params_b: Some(0.494),
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: false,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
        });
        cfg.servers.push(ServerDef {
            backend: "vllm".into(),
            id: "it-bge".into(),
            name: "bge-small".into(),
            model_id: "BAAI/bge-small-en-v1.5".into(),
            task: "embed".into(),
            port: 8131,
            gpu_mem_util: 0.15,
            max_model_len: Some(512),
            quant: "fp16".into(),
            served_model_name: Some("embedder".into()),
            kv_cache_dtype: None,
            llamacpp_channel: crate::state::LlamaCppChannel::Upstream,
            enforce_eager: true,
            params_b: Some(0.033),
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: false,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
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
        let pidline = wsl::run_script(
            DISTRO,
            &format!("cat ~/llm-lp/run/{id}.pid 2>/dev/null || true"),
        )
        .stdout;
        let pid: Option<u32> = pidline.trim().parse().ok();
        println!("{id} pidfile: '{pidline:?}'");
        server::stop_server(&state, None, id).expect("stop");
        if let Some(pid) = pid {
            std::thread::sleep(Duration::from_millis(1500));
            let alive = wsl::run_script(
                DISTRO,
                &format!("kill -0 {pid} 2>/dev/null && echo alive || echo gone"),
            );
            println!("{id} pid {pid}: {}", alive.stdout.trim());
            assert_eq!(
                alive.stdout.trim(),
                "gone",
                "{id} pid still running after stop"
            );
        }
    }

    println!("WSL INTEGRATION OK ✓");
}
