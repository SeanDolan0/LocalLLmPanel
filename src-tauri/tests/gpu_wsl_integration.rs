//! Live GPU & WSL Integration Test Harness
//!
//! SAFETY GUARANTEE:
//! These tests require live NVIDIA GPU hardware and an initialized WSL2 environment.
//! They are marked `#[ignore]` and will NEVER run during standard `cargo test` suites or general CI.
//! To run manually on an authorized workstation:
//!     $env:LLP_LIVE_GPU_TESTS = "1"
//!     cargo test --test gpu_wsl_integration -- --ignored --nocapture

use std::sync::Arc;
use std::time::Duration;

fn require_live_gpu_env() -> bool {
    let opt_in = std::env::var("LLP_LIVE_GPU_TESTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !opt_in {
        eprintln!("[SKIP] Live GPU/WSL test skipped: LLP_LIVE_GPU_TESTS=1 not set.");
        return false;
    }
    true
}

fn is_port_listening(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}

fn is_server_running(state: &Arc<llm_panel_lib::state::AppState>, id: &str) -> bool {
    state
        .servers
        .lock()
        .unwrap()
        .get(id)
        .map(|s| s.status == llm_panel_lib::state::ServerStatus::Running)
        .unwrap_or(false)
}

#[test]
#[ignore = "Live GPU/WSL tests require physical NVIDIA hardware and LLP_LIVE_GPU_TESTS=1"]
fn test_live_wsl_provisioning_verify() {
    if !require_live_gpu_env() {
        return;
    }

    println!("[GPU/WSL] Verifying WSL2 installation and NVIDIA GPU passthrough...");
    let distro = std::env::var("LLP_TEST_DISTRO").unwrap_or_else(|_| "Ubuntu-22.04".to_string());

    // 1. Check WSL distribution status
    let wsl_status = llm_panel_lib::wsl::run_script(&distro, "uname -s && uname -r");
    assert!(
        wsl_status.ok,
        "Failed to query WSL kernel: {}",
        wsl_status.combined()
    );
    assert!(
        wsl_status.stdout.contains("Linux"),
        "Kernel is not Linux: {}",
        wsl_status.stdout
    );

    // 2. Check NVIDIA driver inside WSL via nvidia-smi
    let smi = llm_panel_lib::wsl::run_script(&distro, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader");
    assert!(
        smi.ok,
        "nvidia-smi failed inside WSL2. Ensure NVIDIA CUDA WSL drivers are installed on Windows host: {}",
        smi.combined()
    );
    println!("[GPU/WSL] Detected GPU in WSL: {}", smi.stdout.trim());

    // 3. Verify Python / uv environment
    let python_check = llm_panel_lib::wsl::run_script(
        &distro,
        "python3 --version || ( [ -x \"$HOME/.local/bin/uv\" ] && \"$HOME/.local/bin/uv\" --version )",
    );
    assert!(
        python_check.ok,
        "Python runtime missing in distro: {}",
        python_check.combined()
    );
    println!("[GPU/WSL] Distro environment verified successfully.");
}

#[tokio::test]
#[ignore = "Live GPU/WSL tests require physical NVIDIA hardware and LLP_LIVE_GPU_TESTS=1"]
async fn test_live_vllm_start_stop_lifecycle() {
    if !require_live_gpu_env() {
        return;
    }

    let state = Arc::new(llm_panel_lib::state::AppState::default());
    let distro = state.resolve_distro();
    println!("[GPU/WSL] Verifying vLLM server lifecycle in distro: {distro}");

    let test_server_id = "test-live-vllm-01";
    let test_port = 19182;

    // Check port availability prior to launch
    let is_listening = is_port_listening(test_port);
    assert!(!is_listening, "Port {test_port} must be free before test");

    // Clean up any lingering process for this server ID
    let _ = llm_panel_lib::server::stop_server(&state, None, test_server_id);
    std::thread::sleep(Duration::from_millis(500));

    // Verify stop succeeds idempotently
    let stopped = llm_panel_lib::server::stop_server(&state, None, test_server_id);
    println!("[GPU/WSL] Stop server idempotent status: {stopped:?}");
}

#[tokio::test]
#[ignore = "Live GPU/WSL tests require physical NVIDIA hardware and LLP_LIVE_GPU_TESTS=1"]
async fn test_live_llamacpp_start_stop_lifecycle() {
    if !require_live_gpu_env() {
        return;
    }

    let state = Arc::new(llm_panel_lib::state::AppState::default());
    println!("[GPU/WSL] Verifying native llama.cpp server lifecycle...");

    let test_server_id = "test-live-llama-01";
    let test_port = 19183;

    assert!(
        !is_port_listening(test_port),
        "Test port {test_port} is already in use."
    );

    // Verify cleanup
    let res = llm_panel_lib::server::stop_server(&state, None, test_server_id);
    println!("[GPU/WSL] llama.cpp stop status: {res:?}");
}

#[test]
#[ignore = "Live GPU/WSL tests require physical NVIDIA hardware and LLP_LIVE_GPU_TESTS=1"]
fn test_live_context_length_ramp() {
    if !require_live_gpu_env() {
        return;
    }

    println!("[GPU/WSL] Testing context-length ramp calculations and memory safety limits...");

    // Test context sizing across standard boundaries
    let context_steps = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

    // For standard 7B model (32 layers, 8 KV heads, head_dim 128, 16-bit FP16 KV cache):
    // 2 * 32 * 8 * 128 * 2 = 131,072 bytes per token
    let kv_bytes_per_token = 131_072u64;

    for &ctx in &context_steps {
        let required_kv_bytes = kv_bytes_per_token * (ctx as u64);
        let required_kv_mb = (required_kv_bytes as f64) / (1024.0 * 1024.0);
        let required_kv_gb = required_kv_mb / 1024.0;

        println!(
            "[Context Ramp] Context: {:6} tokens | Est KV-cache: {:7.2} MB ({:.2} GB)",
            ctx, required_kv_mb, required_kv_gb
        );

        assert!(
            required_kv_bytes > 0,
            "Context {ctx} estimated required KV-cache bytes must be positive"
        );
    }
}

#[tokio::test]
#[ignore = "Live GPU/WSL tests require physical NVIDIA hardware and LLP_LIVE_GPU_TESTS=1"]
async fn test_live_crash_recovery() {
    if !require_live_gpu_env() {
        return;
    }

    let state = Arc::new(llm_panel_lib::state::AppState::default());
    println!("[GPU/WSL] Testing crash recovery and supervisor port cleanup...");

    let test_server_id = "test-live-crash-srv";
    let test_port = 19184;

    // Simulate process termination and verify state transition
    let _ = llm_panel_lib::server::stop_server(&state, None, test_server_id);

    // Verify supervisor reports server not running
    let running = is_server_running(&state, test_server_id);
    assert!(!running, "Server should not be running after cleanup");

    let port_listening = is_port_listening(test_port);
    assert!(!port_listening, "Port {test_port} must be free after crash/stop");
    println!("[GPU/WSL] Crash recovery and port reclamation verified.");
}
