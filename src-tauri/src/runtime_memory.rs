//! Runtime model-placement telemetry.
//!
//! Static sizing (`hardware_sizer`) can only estimate where weights will land.
//! llama.cpp itself knows the truth: at `-lv 4` it prints which layers it
//! offloaded and how much of the model buffer stayed in host memory. This
//! module parses those lines from the server log so the UI can warn when a
//! model is (fully or partially) resident in system RAM, even when the model
//! nominally fits in VRAM and the estimate-based spill warning stays quiet.

use serde::{Deserialize, Serialize};

/// Placement facts recovered from a running backend's startup log.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeMemoryReport {
    pub server_id: String,
    pub server_name: String,
    pub backend: String,
    /// Layers the backend placed on the GPU (`offloaded X/Y layers to GPU`).
    pub layers_on_gpu: Option<u32>,
    pub layers_total: Option<u32>,
    /// Model buffers held in device memory (sum of `CUDA0 model buffer size` etc.).
    pub gpu_model_mb: Option<f64>,
    /// Model buffers kept in host memory (`CPU` / `CPU_Mapped model buffer size`).
    pub host_model_mb: Option<f64>,
    /// KV cache buffers in host memory (`CPU KV buffer size`).
    pub host_kv_mb: Option<f64>,
    /// KV cache buffers in GPU memory (`CUDA0 KV buffer size`).
    pub gpu_kv_mb: Option<f64>,
    /// Recurrent / compute / output buffers in host memory.
    pub host_compute_mb: Option<f64>,
    /// Compute buffers in GPU memory.
    pub gpu_compute_mb: Option<f64>,
    /// Total memory in host RAM (sum of host model, kv, and compute buffers).
    pub host_total_mb: Option<f64>,
    /// Total memory in GPU VRAM (sum of gpu model, kv, and compute buffers).
    pub gpu_total_mb: Option<f64>,
    /// Device free memory reported while the model was loaded.
    pub vram_free_mb: Option<u64>,
    /// Free device memory llama.cpp's fit pass intended to leave.
    pub fit_leftover_mb: Option<f64>,
    /// Resident working set of the native backend process.
    pub process_ram_mb: Option<u64>,
    /// True once placement lines were parsed from the backend log.
    pub placement_known: bool,
    /// The user explicitly asked for CPU-only llama.cpp (`-ngl 0`).
    #[serde(default)]
    pub cpu_only_requested: bool,
    /// The one-time console notice has already been emitted.
    #[serde(skip)]
    pub warned: bool,
    /// Number of `loading model` sections observed; only the first (main model)
    /// is parsed so a speculative draft model cannot overwrite main-model data.
    #[serde(skip)]
    section: u32,
}

impl RuntimeMemoryReport {
    pub fn new(server_id: &str, server_name: &str, backend: &str) -> Self {
        Self {
            server_id: server_id.to_string(),
            server_name: server_name.to_string(),
            backend: backend.to_string(),
            ..Self::default()
        }
    }

    pub fn update_totals(&mut self) {
        let host = self.host_model_mb.unwrap_or(0.0)
            + self.host_kv_mb.unwrap_or(0.0)
            + self.host_compute_mb.unwrap_or(0.0);
        self.host_total_mb = if host > 0.0 { Some(host) } else { None };

        let gpu = self.gpu_model_mb.unwrap_or(0.0)
            + self.gpu_kv_mb.unwrap_or(0.0)
            + self.gpu_compute_mb.unwrap_or(0.0);
        self.gpu_total_mb = if gpu > 0.0 { Some(gpu) } else { None };
    }

    /// Parse one raw backend log line. Returns a `[LocalLLmPanel]` notice the
    /// first time a host-memory spill is confirmed.
    pub fn observe_log_line(&mut self, raw: &str) -> Option<String> {
        let line = strip_ansi(raw);
        if line.contains("loading model '") || line.contains("loading model \"") {
            self.section += 1;
            // A second (draft) model begins loading: stop touching the report.
            return None;
        }
        if self.section > 1 {
            return None;
        }

        if let Some((on_gpu, total)) = parse_offloaded_layers(&line) {
            // Only trust the first placement line: it belongs to the main model.
            if self.layers_total.is_none() {
                self.layers_on_gpu = Some(on_gpu);
                self.layers_total = Some(total);
                self.placement_known = true;
            }
        }

        if let Some((device, kind, mb)) = parse_named_buffer(&line) {
            let upper_dev = device.to_ascii_uppercase();
            let upper_kind = kind.to_ascii_uppercase();
            if upper_dev.starts_with("CPU") || upper_dev.contains("HOST") {
                match upper_kind.as_str() {
                    "MODEL" => {
                        self.host_model_mb = Some(self.host_model_mb.unwrap_or(0.0) + mb);
                    }
                    "KV" => {
                        self.host_kv_mb = Some(self.host_kv_mb.unwrap_or(0.0) + mb);
                    }
                    _ => {
                        self.host_compute_mb = Some(self.host_compute_mb.unwrap_or(0.0) + mb);
                    }
                }
            } else if is_device_buffer(&upper_dev) {
                match upper_kind.as_str() {
                    "MODEL" => {
                        self.gpu_model_mb = Some(self.gpu_model_mb.unwrap_or(0.0) + mb);
                    }
                    "KV" => {
                        self.gpu_kv_mb = Some(self.gpu_kv_mb.unwrap_or(0.0) + mb);
                    }
                    _ => {
                        self.gpu_compute_mb = Some(self.gpu_compute_mb.unwrap_or(0.0) + mb);
                    }
                }
            }
            self.update_totals();
        }

        if line.contains("using device") {
            if let Some(free) = parse_trailing_size_mb(&line, " - ") {
                self.vram_free_mb = Some(free.round() as u64);
            }
        }

        if let Some(left) = parse_size_after(&line, "will leave ") {
            self.fit_leftover_mb = Some(left);
        } else if let Some(free) = parse_size_after(&line, "vs. ") {
            if self.vram_free_mb.is_none() {
                self.vram_free_mb = Some(free.round() as u64);
            }
        }

        // Trigger spill notice once host buffers are confirmed, or when server
        // starts serving requests if buffer lines were omitted by the backend.
        let server_ready = line.contains("HTTP server listening")
            || line.contains("server is listening")
            || line.contains("all slots are idle");

        if !self.warned && (self.has_ram_spill() || (server_ready && self.fallback_spill_detected())) {
            self.warned = true;
            return Some(self.spill_notice());
        }
        None
    }

    /// Weights are (at least partly) in system RAM.
    ///
    /// llama.cpp's layer placement is authoritative, so a partial offload
    /// (`10/66 layers`) is a spill even if the model nominally fits in VRAM.
    /// Explicit CPU-only launches are treated as intentional and stay quiet.
    pub fn spill_detected(&self) -> bool {
        if !self.placement_known {
            return false;
        }
        let (Some(on_gpu), Some(total)) = (self.layers_on_gpu, self.layers_total) else {
            return false;
        };
        if total == 0 || on_gpu >= total {
            return false;
        }
        if on_gpu == 0 && self.cpu_only_requested {
            return false;
        }
        match self.host_model_mb {
            Some(mb) => mb >= 64.0,
            None => false,
        }
    }

    pub fn kv_spill_detected(&self) -> bool {
        if self.cpu_only_requested {
            return false;
        }
        match self.host_kv_mb {
            Some(mb) => mb >= 16.0,
            None => false,
        }
    }

    pub fn has_ram_spill(&self) -> bool {
        self.spill_detected() || self.kv_spill_detected()
    }

    fn fallback_spill_detected(&self) -> bool {
        if !self.placement_known {
            return false;
        }
        let (Some(on_gpu), Some(total)) = (self.layers_on_gpu, self.layers_total) else {
            return false;
        };
        if total == 0 || on_gpu >= total {
            return false;
        }
        if on_gpu == 0 && self.cpu_only_requested {
            return false;
        }
        true
    }

    /// Human-readable summary of where the weights went.
    pub fn spill_notice(&self) -> String {
        let mut parts = Vec::new();
        if let (Some(on_gpu), Some(total)) = (self.layers_on_gpu, self.layers_total) {
            parts.push(format!("{on_gpu}/{total} layers on GPU"));
        }
        if let Some(mb) = self.host_model_mb {
            parts.push(format!("~{} model weights in system RAM", fmt_mb(mb)));
        }
        let extra_host = self.host_kv_mb.unwrap_or(0.0) + self.host_compute_mb.unwrap_or(0.0);
        if extra_host >= 16.0 {
            parts.push(format!("~{} context/buffers in system RAM", fmt_mb(extra_host)));
        }
        if let Some(total) = self.host_total_mb {
            if total >= 64.0 {
                parts.push(format!("~{} total RAM used", fmt_mb(total)));
            }
        }
        if let Some(mb) = self.gpu_model_mb {
            parts.push(format!("~{} on GPU", fmt_mb(mb)));
        }
        if let Some(free) = self.vram_free_mb {
            parts.push(format!("{free} MiB VRAM was free at load"));
        }
        format!(
            "[LocalLLmPanel] Model weights/buffers loaded into system RAM ({}).",
            parts.join(", ")
        )
    }
}

fn fmt_mb(mb: f64) -> String {
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{:.0} MB", mb)
    }
}

fn is_device_buffer(device_upper: &str) -> bool {
    const PREFIXES: [&str; 8] = [
        "CUDA", "METAL", "VULKAN", "SYCL", "GPU", "RPC", "ROCM", "HIP",
    ];
    PREFIXES
        .iter()
        .any(|prefix| device_upper.starts_with(prefix))
}

/// Remove ANSI SGR/style escapes (`\x1b[1m`, `\x1b[0m`, …) that llama.cpp
/// emits when it writes to a terminal-like stream.
pub fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// `offloaded 10/66 layers to GPU` → `(10, 66)`.
/// Also matches the older `llm_load_tensors:` prefix because the phrase itself
/// is the same.
fn parse_offloaded_layers(line: &str) -> Option<(u32, u32)> {
    let marker = "offloaded ";
    let idx = line.find(marker)? + marker.len();
    let rest = &line[idx..];
    let slash = rest.find('/')?;
    let on_gpu = rest[..slash].trim().parse::<u32>().ok()?;
    let after = &rest[slash + 1..];
    let end = after
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(after.len());
    let total = after[..end].parse::<u32>().ok()?;
    if total == 0 {
        return None;
    }
    Some((on_gpu, total))
}

/// Matches lines like:
/// `load_tensors:   CPU_Mapped model buffer size =   640.00 MiB` -> `("CPU_Mapped", "model", 640.0)`
/// `llama_kv_cache: CPU KV buffer size =    28.00 MiB` -> `("CPU", "kv", 28.0)`
/// `sched_reserve:  CUDA_Host compute buffer size =    41.00 MiB` -> `("CUDA_Host", "compute", 41.0)`
/// `sched_reserve:      CUDA0 compute buffer size =   216.29 MiB` -> `("CUDA0", "compute", 216.29)`
/// `llama_context:  CUDA_Host  output buffer size =     3.79 MiB` -> `("CUDA_Host", "output", 3.79)`
/// `llama_memory_recurrent:        CPU RS buffer size =   523.69 MiB` -> `("CPU", "rs", 523.69)`
fn parse_named_buffer(line: &str) -> Option<(&str, &str, f64)> {
    let idx = line.find(" buffer size")?;
    let prefix = line[..idx].trim_end();
    let mut words = prefix.split_whitespace().rev();
    let kind_or_dev = words.next()?;
    let (device, kind) = match kind_or_dev.to_ascii_uppercase().as_str() {
        "MODEL" | "KV" | "RS" | "COMPUTE" | "OUTPUT" => {
            let dev = words.next().unwrap_or(kind_or_dev);
            (dev, kind_or_dev)
        }
        _ => (kind_or_dev, "model"),
    };
    let rest = line[idx + " buffer size".len()..].trim_start();
    let rest = rest.trim_start_matches(|ch: char| ch == '=' || ch.is_whitespace());
    let mb = parse_size_prefix(rest)?;
    Some((device, kind, mb))
}

#[allow(dead_code)]
fn parse_model_buffer(line: &str) -> Option<(&str, f64)> {
    parse_named_buffer(line).map(|(dev, _, mb)| (dev, mb))
}

/// Parse a leading `"<number> <unit>"` pair into MiB.
fn parse_size_prefix(value: &str) -> Option<f64> {
    let mut parts = value.split_whitespace();
    let number: f64 = parts.next()?.trim_end_matches(',').parse().ok()?;
    let unit = parts.next().unwrap_or("MiB").to_ascii_lowercase();
    let mb = match unit.as_str() {
        "gib" | "gb" => number * 1024.0,
        "mib" | "mb" => number,
        "kib" | "kb" => number / 1024.0,
        "b" | "bytes" => number / (1024.0 * 1024.0),
        // `will leave 8149 >= 1024 MiB ...`: bare numbers are MiB.
        _ => number,
    };
    Some(mb)
}

/// Parse a size that follows `marker` in the line, ignoring trailing prose.
fn parse_size_after(line: &str, marker: &str) -> Option<f64> {
    let idx = line.find(marker)? + marker.len();
    parse_size_prefix(line[idx..].trim_start())
}

/// Parse a size following the *last* occurrence of `marker`.
fn parse_trailing_size_mb(line: &str, marker: &str) -> Option<f64> {
    let idx = line.rfind(marker)? + marker.len();
    parse_size_prefix(line[idx..].trim_start())
}

/// Resident working set of `pid` in MiB.
#[cfg(windows)]
pub fn process_working_set_mb(pid: u32) -> Option<u64> {
    use std::ffi::c_void;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const PROCESS_VM_READ: u32 = 0x0010;

    unsafe {
        let handle = OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
            0,
            pid,
        );
        if handle.is_null() {
            return None;
        }
        let mut counters: ProcessMemoryCounters = std::mem::zeroed();
        counters.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
        let ok = K32GetProcessMemoryInfo(handle, &mut counters, counters.cb);
        CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        Some((counters.WorkingSetSize as u64) / (1024 * 1024))
    }
}

#[cfg(not(windows))]
pub fn process_working_set_mb(_pid: u32) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> RuntimeMemoryReport {
        RuntimeMemoryReport::new("srv-1", "test", "llamacpp")
    }

    #[test]
    fn detects_partial_offload() {
        let mut r = report();
        let lines = [
            "0.03.067.887 I load_tensors: offloading output layer to GPU",
            "0.03.067.892 I load_tensors: offloading 9 repeating layers to GPU",
            "0.03.067.892 I load_tensors: offloaded 10/66 layers to GPU",
            "0.03.067.896 I load_tensors:   CPU_Mapped model buffer size =  6263.93 MiB",
            "0.03.067.899 I load_tensors:        CUDA0 model buffer size =  1758.23 MiB",
            "0.00.841.264 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU) (0000:02:00.0) - 11042 MiB free",
        ];
        let mut notice = None;
        for line in lines {
            notice = r.observe_log_line(line).or(notice);
        }
        assert_eq!(r.layers_on_gpu, Some(10));
        assert_eq!(r.layers_total, Some(66));
        assert!((r.host_model_mb.unwrap() - 6263.93).abs() < 0.01);
        assert!((r.gpu_model_mb.unwrap() - 1758.23).abs() < 0.01);
        assert_eq!(r.vram_free_mb, Some(11042));
        assert!(r.spill_detected());
        assert!(notice.unwrap().contains("10/66 layers on GPU"));
        assert!(notice_contains_gb(&r));
    }

    fn notice_contains_gb(r: &RuntimeMemoryReport) -> bool {
        r.spill_notice().contains("6.1 GB")
    }

    #[test]
    fn full_offload_is_not_a_spill() {
        let mut r = report();
        for line in [
            "0.01.259.985 I load_tensors: offloaded 37/37 layers to GPU",
            "0.01.259.990 I load_tensors:   CPU_Mapped model buffer size =   640.00 MiB",
            "0.01.259.991 I load_tensors:        CUDA0 model buffer size =  2734.15 MiB",
        ] {
            assert!(r.observe_log_line(line).is_none());
        }
        assert!(r.placement_known);
        assert!(!r.spill_detected());
    }

    #[test]
    fn older_log_format_is_parsed() {
        let mut r = report();
        r.observe_log_line("llama_model_load: loading model");
        r.observe_log_line("llm_load_tensors: offloaded 5/33 layers to GPU");
        r.observe_log_line("llm_load_tensors: CPU model buffer size =  2048.00 MiB");
        r.observe_log_line("llm_load_tensors: CUDA0 model buffer size =  1024.00 MiB");
        assert_eq!(r.layers_on_gpu, Some(5));
        assert_eq!(r.layers_total, Some(33));
        assert!((r.host_model_mb.unwrap() - 2048.0).abs() < 0.01);
        assert!(r.spill_detected());
    }

    #[test]
    fn ansi_escapes_are_stripped() {
        let mut r = report();
        r.observe_log_line("\u{1b}[1mload_tensors:\u{1b}[0m offloaded 2/4 layers to GPU");
        assert_eq!(r.layers_on_gpu, Some(2));
        assert_eq!(r.layers_total, Some(4));
        assert_eq!(strip_ansi("\u{1b}[1mhi\u{1b}[0m"), "hi");
    }

    #[test]
    fn fit_lines_are_recorded() {
        let mut r = report();
        r.observe_log_line(
            "0.00.982.010 I common_params_fit_impl: projected to use 2892 MiB of device memory vs. 11042 MiB of free device memory",
        );
        assert_eq!(r.vram_free_mb, Some(11042));
        r.observe_log_line(
            "0.00.982.016 I common_params_fit_impl: will leave 8149 >= 1024 MiB of free device memory, no changes needed",
        );
        assert_eq!(r.fit_leftover_mb, Some(8149.0));
    }

    #[test]
    fn second_load_section_is_ignored() {
        let mut r = report();
        r.observe_log_line("0.01.000.000 I srv    load_model: loading model 'main.gguf'");
        r.observe_log_line("0.01.100.000 I load_tensors: offloaded 8/64 layers to GPU");
        r.observe_log_line("0.01.100.100 I load_tensors:   CPU_Mapped model buffer size =  4096.00 MiB");
        r.observe_log_line("0.02.000.000 I srv    load_model: loading model 'draft.gguf'");
        r.observe_log_line("0.02.100.000 I load_tensors: offloaded 36/36 layers to GPU");
        r.observe_log_line("0.02.100.100 I load_tensors:        CUDA0 model buffer size =   500.00 MiB");
        assert_eq!(r.layers_on_gpu, Some(8));
        assert_eq!(r.layers_total, Some(64));
        assert!((r.host_model_mb.unwrap() - 4096.0).abs() < 0.01);
        assert!(r.gpu_model_mb.is_none());
    }

    #[test]
    fn cpu_only_can_be_intentional() {
        let mut r = report();
        r.cpu_only_requested = true;
        r.observe_log_line("load_tensors: offloaded 0/64 layers to GPU");
        r.observe_log_line("load_tensors: CPU model buffer size =  8192.00 MiB");
        assert!(!r.spill_detected());

        let mut forced = report();
        forced.observe_log_line("load_tensors: offloaded 0/64 layers to GPU");
        forced.observe_log_line("load_tensors: CPU model buffer size =  8192.00 MiB");
        assert!(forced.spill_detected());
    }

    #[test]
    fn notice_emitted_once() {
        let mut r = report();
        r.observe_log_line("offloaded 1/64 layers to GPU");
        let first = r.observe_log_line("CPU_Mapped model buffer size = 3000 MiB");
        assert!(first.is_some());
        assert!(r.observe_log_line("CUDA0 model buffer size = 100 MiB").is_none());
    }

    #[test]
    fn kv_and_compute_buffers_are_parsed() {
        let mut r = report();
        let lines = [
            "0.03.067.892 I load_tensors: offloaded 10/66 layers to GPU",
            "0.03.067.896 I load_tensors:   CPU_Mapped model buffer size =  6263.93 MiB",
            "0.03.067.899 I load_tensors:        CUDA0 model buffer size =  1758.23 MiB",
            "0.03.799.216 I llama_context:  CUDA_Host  output buffer size =     3.79 MiB",
            "0.03.805.820 I llama_kv_cache:        CPU KV buffer size =    28.00 MiB",
            "0.03.806.148 I llama_kv_cache:      CUDA0 KV buffer size =     4.00 MiB",
            "0.03.918.450 I llama_memory_recurrent:        CPU RS buffer size =   523.69 MiB",
            "0.03.920.244 I llama_memory_recurrent:      CUDA0 RS buffer size =    74.81 MiB",
            "0.03.941.577 I sched_reserve:      CUDA0 compute buffer size =   216.29 MiB",
            "0.03.941.583 I sched_reserve:  CUDA_Host compute buffer size =    41.00 MiB",
        ];
        for line in lines {
            r.observe_log_line(line);
        }
        assert_eq!(r.host_kv_mb, Some(28.0));
        assert_eq!(r.gpu_kv_mb, Some(4.0));
        assert!((r.host_compute_mb.unwrap() - 568.48).abs() < 0.01);
        assert!((r.host_total_mb.unwrap() - 6860.41).abs() < 0.01);
        assert!(r.has_ram_spill());
    }

    #[cfg(windows)]
    #[test]
    fn own_working_set_is_readable() {
        let mb = process_working_set_mb(std::process::id()).expect("working set");
        assert!(mb > 0);
    }
}
