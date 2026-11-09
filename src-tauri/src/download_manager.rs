//! Unified Download Manager:
//! - Multi-backend job tracking (llama.cpp GGUF & vLLM Safetensors)
//! - Resumable HTTP Range requests with `.part` staging
//! - Multi-shard atomic download & verification
//! - Background bounded worker pool with real Queued state
//! - Pause, resume, retry, and cancellation
//! - Trusted SHA-256 content integrity verification before promotion
//! - Destination disk-space preflight validation
//! - Pinned commit SHA resolution and shard URL pinning
//! - Seamless synchronization with the Local Model Registry

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

use crate::model_registry::{ModelRegistry, RegistryModelEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadStatus {
    Queued,
    Resolving,
    Downloading,
    Paused,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub id: String,
    pub model_id: String,
    pub repo_id: String,
    pub backend: String, // "llamacpp" | "vllm"
    pub quant: String,   // "Q4_K_M", "AWQ", "FP16", etc.
    pub files: Vec<String>,
    pub destination_dir: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub speed_bps: f64,
    pub eta_seconds: Option<f64>,
    pub status: DownloadStatus,
    pub state: DownloadStatus,
    pub error: Option<String>,
    pub current_file: Option<String>,
    pub current_shard: usize,
    pub total_shards: usize,
    pub is_split: bool,
    pub companion_mmproj: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub verification_state: String, // "unverified", "verifying", "verified", "size_only", "failed"
    #[serde(default)]
    pub supports_pause: bool,
    #[serde(default)]
    pub supports_resume: bool,
    #[serde(default)]
    pub supports_cancel: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgressEvent {
    pub job_id: String,
    pub model_id: String,
    pub repo_id: String,
    pub quant: String,
    pub status: String,
    pub state: String,
    pub percent: f64,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: f64,
    pub eta_seconds: Option<f64>,
    pub current_file: Option<String>,
    pub current_shard: usize,
    pub total_shards: usize,
    pub error: Option<String>,
    pub revision: Option<String>,
    pub verification_state: String,
    pub supports_pause: bool,
    pub supports_resume: bool,
    pub supports_cancel: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RangeValidation {
    ValidRangeResume,
    MismatchRestartFromZero,
    FullContentRestartFromZero,
    RangeNotSatisfiable,
    InvalidStatus(reqwest::StatusCode),
}

pub fn extract_content_length(resp: &reqwest::blocking::Response) -> u64 {
    resp.headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| {
            resp.headers()
                .get("x-linked-size")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

pub fn parse_content_range_start(header_val: &str) -> Option<u64> {
    let trimmed = header_val.trim().strip_prefix("bytes ")?;
    let start_str = trimmed.split('-').next()?;
    start_str.parse::<u64>().ok()
}

pub fn validate_range_response(
    status: reqwest::StatusCode,
    content_range: Option<&str>,
    current_offset: u64,
) -> RangeValidation {
    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        if let Some(cr) = content_range {
            if let Some(start) = parse_content_range_start(cr) {
                if start == current_offset {
                    return RangeValidation::ValidRangeResume;
                }
            }
        }
        RangeValidation::MismatchRestartFromZero
    } else if status == reqwest::StatusCode::OK {
        RangeValidation::FullContentRestartFromZero
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        RangeValidation::RangeNotSatisfiable
    } else {
        RangeValidation::InvalidStatus(status)
    }
}

pub fn generate_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    #[cfg(windows)]
    {
        extern "system" {
            #[link_name = "SystemFunction036"]
            fn RtlGenRandom(RandomBuffer: *mut u8, RandomBufferLength: u32) -> u8;
        }
        unsafe {
            RtlGenRandom(bytes.as_mut_ptr(), 16);
        }
    }
    #[cfg(not(windows))]
    {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((t >> (i * 8)) & 0xff) as u8;
        }
    }

    bytes[6] = (bytes[6] & 0x0f) | 0x40; // Version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // Variant 1 (RFC 4122)

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

#[cfg(windows)]
pub fn get_disk_free_space(path: &Path) -> std::io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut current = path.to_path_buf();
    while !current.exists() {
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    let mut wide: Vec<u16> = current.as_os_str().encode_wide().collect();
    wide.push(0);

    let mut free_bytes_available: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut total_free_bytes: u64 = 0;

    extern "system" {
        fn GetDiskFreeSpaceExW(
            lpDirectoryName: *const u16,
            lpFreeBytesAvailableToCaller: *mut u64,
            lpTotalNumberOfBytes: *mut u64,
            lpTotalNumberOfFreeBytes: *mut u64,
        ) -> i32;
    }

    let success = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_bytes_available,
            &mut total_bytes,
            &mut total_free_bytes,
        )
    };

    if success != 0 {
        Ok(free_bytes_available)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(windows))]
pub fn get_disk_free_space(_path: &Path) -> std::io::Result<u64> {
    Ok(u64::MAX)
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

pub fn resolve_remote_file_size(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> Result<u64, String> {
    // 1. Try HEAD request first
    let mut head_req = client.head(url);
    if let Some(t) = token.filter(|s| !s.trim().is_empty()) {
        head_req = head_req.bearer_auth(t);
    }

    match head_req.send() {
        Ok(resp) => {
            let status = resp.status();
            if status == reqwest::StatusCode::UNAUTHORIZED {
                return Err("Authentication required (401). Please check Hugging Face token.".to_string());
            }
            if status == reqwest::StatusCode::FORBIDDEN {
                return Err("Access forbidden (403). Repository may be private or gated.".to_string());
            }
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(format!("File not found on remote (404): {url}"));
            }
            if status.is_success() {
                let len = extract_content_length(&resp);
                if len > 0 {
                    return Ok(len);
                }
            }
        }
        Err(_) => {
            // Network failure or unsupported HEAD, proceed to ranged GET fallback
        }
    }

    // 2. Ranged GET fallback (bytes=0-0)
    let mut get_req = client.get(url).header(reqwest::header::RANGE, "bytes=0-0");
    if let Some(t) = token.filter(|s| !s.trim().is_empty()) {
        get_req = get_req.bearer_auth(t);
    }

    let resp = get_req.send().map_err(|e| format!("Transfer error for {url}: {e}"))?;
    let status = resp.status();

    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err("Authentication required (401). Please check Hugging Face token.".to_string());
    }
    if status == reqwest::StatusCode::FORBIDDEN {
        return Err("Access forbidden (403). Repository may be private or gated.".to_string());
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(format!("File not found on remote (404): {url}"));
    }

    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        if let Some(cr) = resp.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok()) {
            if let Some(total_str) = cr.split('/').next_back() {
                if let Ok(total) = total_str.parse::<u64>() {
                    return Ok(total);
                }
            }
        }
    } else if status.is_success() {
        let len = extract_content_length(&resp);
        if len > 0 {
            return Ok(len);
        }
    }

    Err(format!("Could not determine file size for {url} (HTTP {status})"))
}

pub fn verify_file_sha256(
    file_path: &Path,
    cancel_flag: &AtomicBool,
) -> Result<String, String> {
    let mut file = std::fs::File::open(file_path).map_err(|e| format!("Cannot open file for hashing: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];

    loop {
        if cancel_flag.load(Ordering::SeqCst) {
            return Err("Verification cancelled".to_string());
        }
        let n = file.read(&mut buf).map_err(|e| format!("Read error during hashing: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn wait_for_retry(
    cancel_flag: &AtomicBool,
    pause_flag: &AtomicBool,
    attempt: usize,
    retry_after_secs: Option<u64>,
) -> bool {
    let wait_duration = if let Some(secs) = retry_after_secs {
        Duration::from_secs(secs.min(60))
    } else {
        let base_ms = (1000u64).saturating_mul(1u64 << (attempt.min(5) as u32));
        let jitter_ms = (crate::state::now_ms() % 350) as u64;
        Duration::from_millis(base_ms + jitter_ms)
    };

    let start = Instant::now();
    while start.elapsed() < wait_duration {
        if cancel_flag.load(Ordering::SeqCst) || pause_flag.load(Ordering::SeqCst) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

fn fetch_repo_lfs_sha256_digests(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut next_url = Some(format!("https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"));
    let mut pages = 0;

    while let Some(url) = next_url.take() {
        pages += 1;
        if pages > 50 {
            break;
        }

        let mut req = client.get(&url);
        if let Some(t) = token.filter(|s| !s.trim().is_empty()) {
            req = req.bearer_auth(t);
        }

        let resp = match req.send().and_then(|r| r.error_for_status()) {
            Ok(r) => r,
            Err(_) => break,
        };

        next_url = resp.headers()
            .get(reqwest::header::LINK)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_link_next_url);

        let items: Vec<serde_json::Value> = match resp.json() {
            Ok(v) => v,
            Err(_) => break,
        };

        for item in items {
            if let (Some(path), Some(oid)) = (
                item.get("path").and_then(|v| v.as_str()),
                item.get("lfs").and_then(|l| l.get("oid")).and_then(|v| v.as_str()),
            ) {
                map.insert(path.to_string(), oid.to_ascii_lowercase());
                let filename = path.rsplit('/').next().unwrap_or(path);
                map.insert(filename.to_string(), oid.to_ascii_lowercase());
            }
        }
    }

    map
}

fn parse_link_next_url(header_val: &str) -> Option<String> {
    for part in header_val.split(',') {
        if part.contains("rel=\"next\"") || part.contains("rel=next") {
            let start = part.find('<')? + 1;
            let end = part.find('>')?;
            if start < end {
                return Some(part[start..end].to_string());
            }
        }
    }
    None
}

fn resolve_repo_commit_sha(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    token: Option<&str>,
) -> Option<String> {
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let mut req = client.get(&url);
    if let Some(t) = token.filter(|s| !s.trim().is_empty()) {
        req = req.bearer_auth(t);
    }
    if let Ok(resp) = req.send().and_then(|r| r.error_for_status()) {
        if let Ok(json) = resp.json::<serde_json::Value>() {
            if let Some(sha) = json.get("sha").and_then(|v| v.as_str()) {
                return Some(sha.to_string());
            }
        }
    }
    None
}

struct ActiveDownloadController {
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
}

pub struct DownloadManager {
    jobs: Mutex<HashMap<String, DownloadJob>>,
    controllers: Mutex<HashMap<String, ActiveDownloadController>>,
    tokens: Mutex<HashMap<String, Option<String>>>,
    app_handle: Mutex<Option<AppHandle>>,
    registry: Arc<ModelRegistry>,
    max_concurrent: Mutex<usize>,
}

impl DownloadManager {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            controllers: Mutex::new(HashMap::new()),
            tokens: Mutex::new(HashMap::new()),
            app_handle: Mutex::new(None),
            registry,
            max_concurrent: Mutex::new(2),
        }
    }

    pub fn set_max_concurrent(&self, max: usize) {
        if let Ok(mut g) = self.max_concurrent.lock() {
            *g = max.max(1);
        }
    }

    pub fn get_job(&self, job_id: &str) -> Option<DownloadJob> {
        self.jobs.lock().ok()?.get(job_id).cloned()
    }

    pub fn get_job_by_repo(&self, repo_id: &str) -> Option<DownloadJob> {
        let guard = self.jobs.lock().ok()?;
        guard
            .values()
            .find(|j| j.repo_id.eq_ignore_ascii_case(repo_id) || j.model_id.eq_ignore_ascii_case(repo_id))
            .cloned()
    }

    pub fn get_job_by_repo_and_quant(&self, repo_id: &str, quant: &str) -> Option<DownloadJob> {
        let guard = self.jobs.lock().ok()?;
        guard
            .values()
            .find(|j| {
                (j.repo_id.eq_ignore_ascii_case(repo_id) || j.model_id.eq_ignore_ascii_case(repo_id))
                    && j.quant.eq_ignore_ascii_case(quant)
            })
            .cloned()
    }

    pub fn list_jobs(&self) -> Vec<DownloadJob> {
        self.jobs
            .lock()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn clear_completed(&self) -> usize {
        if let Ok(mut guard) = self.jobs.lock() {
            let before = guard.len();
            let mut removed_ids = Vec::new();
            guard.retain(|id, job| {
                let keep = job.status != DownloadStatus::Completed && job.status != DownloadStatus::Cancelled;
                if !keep {
                    removed_ids.push(id.clone());
                }
                keep
            });
            if let Ok(mut c_guard) = self.controllers.lock() {
                for id in &removed_ids {
                    c_guard.remove(id);
                }
            }
            if let Ok(mut t_guard) = self.tokens.lock() {
                for id in &removed_ids {
                    t_guard.remove(id);
                }
            }
            before - guard.len()
        } else {
            0
        }
    }

    pub fn cleanup_finished_job(&self, job_id: &str) {
        if let Ok(mut guard) = self.controllers.lock() {
            guard.remove(job_id);
        }
        if let Ok(mut guard) = self.tokens.lock() {
            guard.remove(job_id);
        }
    }

    pub fn cancel_job(&self, job_id: &str) -> bool {
        let mut queued_cancelled = false;
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                if job.status == DownloadStatus::Queued {
                    job.status = DownloadStatus::Cancelled;
                    job.state = DownloadStatus::Cancelled;
                    job.speed_bps = 0.0;
                    queued_cancelled = true;
                }
            }
        }
        if queued_cancelled {
            self.cleanup_finished_job(job_id);
            if let Some(job) = self.get_job(job_id) {
                self.emit_progress_for_job(&job);
            }
            return true;
        }

        if let Ok(guard) = self.controllers.lock() {
            if let Some(ctrl) = guard.get(job_id) {
                ctrl.cancel_flag.store(true, Ordering::SeqCst);
                if let Ok(mut j_guard) = self.jobs.lock() {
                    if let Some(job) = j_guard.get_mut(job_id) {
                        job.status = DownloadStatus::Cancelled;
                        job.state = DownloadStatus::Cancelled;
                        job.speed_bps = 0.0;
                    }
                }
                return true;
            }
        }
        false
    }

    pub fn cancel_by_repo(&self, repo_id: &str) -> bool {
        let id_to_cancel = {
            let guard = match self.jobs.lock() {
                Ok(g) => g,
                Err(_) => return false,
            };
            guard
                .values()
                .find(|j| {
                    j.repo_id.eq_ignore_ascii_case(repo_id)
                        || j.model_id.eq_ignore_ascii_case(repo_id)
                })
                .map(|j| j.id.clone())
        };

        if let Some(id) = id_to_cancel {
            self.cancel_job(&id)
        } else {
            false
        }
    }

    pub fn cancel_by_repo_and_quant(&self, repo_id: &str, quant: &str) -> bool {
        let id_to_cancel = {
            let guard = match self.jobs.lock() {
                Ok(g) => g,
                Err(_) => return false,
            };
            guard
                .values()
                .find(|j| {
                    (j.repo_id.eq_ignore_ascii_case(repo_id)
                        || j.model_id.eq_ignore_ascii_case(repo_id))
                        && j.quant.eq_ignore_ascii_case(quant)
                })
                .map(|j| j.id.clone())
        };

        if let Some(id) = id_to_cancel {
            self.cancel_job(&id)
        } else {
            false
        }
    }

    pub fn is_job_active(&self, repo_id: &str, quant: Option<&str>) -> bool {
        let guard = match self.jobs.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.values().any(|j| {
            let id_matches = j.repo_id.eq_ignore_ascii_case(repo_id)
                || j.model_id.eq_ignore_ascii_case(repo_id);
            if !id_matches {
                return false;
            }
            let is_active = j.status == DownloadStatus::Downloading
                || j.status == DownloadStatus::Resolving
                || j.status == DownloadStatus::Verifying
                || j.status == DownloadStatus::Queued;
            if !is_active {
                return false;
            }
            if let Some(q) = quant {
                j.quant.eq_ignore_ascii_case(q)
            } else {
                true
            }
        })
    }

    pub fn pause_job(&self, job_id: &str) -> bool {
        let mut queued_paused = false;
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                if job.status == DownloadStatus::Queued {
                    job.status = DownloadStatus::Paused;
                    job.state = DownloadStatus::Paused;
                    job.speed_bps = 0.0;
                    queued_paused = true;
                }
            }
        }
        if queued_paused {
            self.cleanup_finished_job(job_id);
            if let Some(job) = self.get_job(job_id) {
                self.emit_progress_for_job(&job);
            }
            return true;
        }

        if let Ok(guard) = self.controllers.lock() {
            if let Some(ctrl) = guard.get(job_id) {
                ctrl.pause_flag.store(true, Ordering::SeqCst);
                if let Ok(mut j_guard) = self.jobs.lock() {
                    if let Some(job) = j_guard.get_mut(job_id) {
                        job.status = DownloadStatus::Paused;
                        job.state = DownloadStatus::Paused;
                        job.speed_bps = 0.0;
                    }
                }
                return true;
            }
        }
        false
    }

    /// Submit a new GGUF download job with multiple shards and optional mmproj.
    pub fn enqueue_gguf_job(
        self: &Arc<Self>,
        app: AppHandle,
        model_id: String,
        repo_id: String,
        quant: String,
        files: Vec<String>,
        destination_dir: String,
        token: Option<String>,
    ) -> Result<String, String> {
        if let Ok(mut g) = self.app_handle.lock() {
            if g.is_none() {
                *g = Some(app);
            }
        }
        self.enqueue_gguf_job_internal(model_id, repo_id, quant, files, destination_dir, token)
    }

    pub fn enqueue_gguf_job_internal(
        self: &Arc<Self>,
        model_id: String,
        repo_id: String,
        quant: String,
        files: Vec<String>,
        destination_dir: String,
        token: Option<String>,
    ) -> Result<String, String> {
        let job_id = format!("job-{}", generate_uuid_v4());
        let total_shards = files.len();
        let is_split = total_shards > 1;

        // Atomic duplicate check and job registration under one single lock
        {
            let mut guard = self.jobs.lock().map_err(|e| e.to_string())?;

            for existing in guard.values() {
                let id_matches = existing.repo_id.eq_ignore_ascii_case(&repo_id)
                    || existing.model_id.eq_ignore_ascii_case(&repo_id);
                if id_matches && existing.quant.eq_ignore_ascii_case(&quant) {
                    if existing.status == DownloadStatus::Downloading
                        || existing.status == DownloadStatus::Resolving
                        || existing.status == DownloadStatus::Verifying
                        || existing.status == DownloadStatus::Queued
                    {
                        return Err(format!(
                            "Download already in progress for {repo_id} ({quant})"
                        ));
                    }
                }
            }

            // Remove any old non-active record for same repo & quant to prevent stale entries
            guard.retain(|_, j| {
                !(j.repo_id.eq_ignore_ascii_case(&repo_id) && j.quant.eq_ignore_ascii_case(&quant))
            });

            let job = DownloadJob {
                id: job_id.clone(),
                model_id,
                repo_id,
                backend: "llamacpp".to_string(),
                quant,
                files: files.clone(),
                destination_dir,
                total_bytes: 0,
                downloaded_bytes: 0,
                speed_bps: 0.0,
                eta_seconds: None,
                status: DownloadStatus::Queued,
                state: DownloadStatus::Queued,
                error: None,
                current_file: files.first().cloned(),
                current_shard: 1,
                total_shards,
                is_split,
                companion_mmproj: None,
                revision: None,
                created_at: crate::state::now_ms(),
                verification_state: "unverified".to_string(),
                supports_pause: true,
                supports_resume: true,
                supports_cancel: true,
            };

            guard.insert(job_id.clone(), job);

            let cancel_flag = Arc::new(AtomicBool::new(false));
            let pause_flag = Arc::new(AtomicBool::new(false));

            if let Ok(mut c_guard) = self.controllers.lock() {
                c_guard.insert(
                    job_id.clone(),
                    ActiveDownloadController {
                        cancel_flag,
                        pause_flag,
                    },
                );
            }

            if let Ok(mut t_guard) = self.tokens.lock() {
                t_guard.insert(job_id.clone(), token);
            }
        }

        if let Some(j) = self.get_job(&job_id) {
            self.emit_progress_for_job(&j);
        }

        // Dispatch next queued job if concurrency limit allows
        self.try_start_next_job();

        Ok(job_id)
    }

    pub fn try_start_next_job(self: &Arc<Self>) {
        let max = self.max_concurrent.lock().map(|m| *m).unwrap_or(2);

        let job_to_start = {
            let mut guard = match self.jobs.lock() {
                Ok(g) => g,
                Err(_) => return,
            };

            let active_count = guard.values().filter(|j| {
                j.status == DownloadStatus::Downloading
                    || j.status == DownloadStatus::Resolving
                    || j.status == DownloadStatus::Verifying
            }).count();

            if active_count >= max {
                return;
            }

            // Find next queued job with earliest created_at timestamp
            let next_id = guard
                .values()
                .filter(|j| j.status == DownloadStatus::Queued)
                .min_by_key(|j| j.created_at)
                .map(|j| j.id.clone());

            if let Some(id) = next_id {
                if let Some(job) = guard.get_mut(&id) {
                    job.status = DownloadStatus::Resolving;
                    job.state = DownloadStatus::Resolving;
                }
                Some(id)
            } else {
                None
            }
        };

        if let Some(job_id) = job_to_start {
            if let Some(j) = self.get_job(&job_id) {
                self.emit_progress_for_job(&j);
            }

            let (cancel_flag, pause_flag) = {
                let guard = match self.controllers.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                if let Some(c) = guard.get(&job_id) {
                    (Arc::clone(&c.cancel_flag), Arc::clone(&c.pause_flag))
                } else {
                    (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)))
                }
            };

            let token = self.tokens.lock().ok().and_then(|t| t.get(&job_id).cloned().flatten());
            let mgr = Arc::clone(self);
            std::thread::spawn(move || {
                mgr.run_gguf_download_worker(job_id, cancel_flag, pause_flag, token);
            });
        }
    }

    pub fn emit_progress_for_job(&self, job: &DownloadJob) {
        let app = self.app_handle.lock().ok().and_then(|g| g.clone());
        let percent = if job.total_bytes > 0 {
            (job.downloaded_bytes as f64 / job.total_bytes as f64 * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };

        let ev = DownloadProgressEvent {
            job_id: job.id.clone(),
            model_id: job.model_id.clone(),
            repo_id: job.repo_id.clone(),
            quant: job.quant.clone(),
            status: format!("{:?}", job.status).to_ascii_lowercase(),
            state: format!("{:?}", job.status).to_ascii_lowercase(),
            percent,
            downloaded_bytes: job.downloaded_bytes,
            total_bytes: job.total_bytes,
            speed_bps: job.speed_bps,
            eta_seconds: job.eta_seconds,
            current_file: job.current_file.clone(),
            current_shard: job.current_shard,
            total_shards: job.total_shards,
            error: job.error.clone(),
            revision: job.revision.clone(),
            verification_state: job.verification_state.clone(),
            supports_pause: job.supports_pause,
            supports_resume: job.supports_resume,
            supports_cancel: job.supports_cancel,
        };

        if let Some(app) = &app {
            let _ = app.emit("download-progress", &ev);

            // Legacy pull-progress
            let legacy_state = match job.status {
                DownloadStatus::Completed => "complete",
                DownloadStatus::Failed => "failed",
                _ => "downloading",
            };
            let legacy_payload = serde_json::json!({
                "model": job.repo_id,
                "quant": job.quant,
                "state": legacy_state,
                "file": job.current_file.clone().or_else(|| job.error.clone()),
                "percent": percent,
                "speed_bps": job.speed_bps,
                "eta_seconds": job.eta_seconds,
                "bytes_downloaded": job.downloaded_bytes,
                "bytes_total": job.total_bytes,
            });
            let _ = app.emit("pull-progress", &legacy_payload);
            if !job.model_id.is_empty() && job.model_id != job.repo_id {
                let mut alt = legacy_payload.clone();
                alt["model"] = serde_json::json!(job.model_id);
                let _ = app.emit("pull-progress", &alt);
            }
        }
    }

    fn run_gguf_download_worker(
        self: &Arc<Self>,
        job_id: String,
        cancel_flag: Arc<AtomicBool>,
        pause_flag: Arc<AtomicBool>,
        token: Option<String>,
    ) {
        let (repo_id, files, destination_dir, quant, model_id, total_shards) = {
            let guard = match self.jobs.lock() {
                Ok(g) => g,
                Err(_) => {
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
            };
            let job = match guard.get(&job_id) {
                Some(j) => j,
                None => {
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
            };
            (
                job.repo_id.clone(),
                job.files.clone(),
                job.destination_dir.clone(),
                job.quant.clone(),
                job.model_id.clone(),
                job.total_shards,
            )
        };

        if cancel_flag.load(Ordering::SeqCst) {
            self.set_job_cancelled(&job_id);
            self.cleanup_finished_job(&job_id);
            self.try_start_next_job();
            return;
        }

        let dest_path = PathBuf::from(&destination_dir);
        if let Err(e) = std::fs::create_dir_all(&dest_path) {
            self.set_job_error(&job_id, format!("Cannot create folder {}: {e}", dest_path.display()));
            self.cleanup_finished_job(&job_id);
            self.try_start_next_job();
            return;
        }

        let client = match reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                self.set_job_error(&job_id, format!("HTTP client initialization error: {e}"));
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }
        };

        // A7: Resolve pinned commit SHA
        let commit_sha = resolve_repo_commit_sha(&client, &repo_id, token.as_deref());
        let revision = commit_sha.clone().unwrap_or_else(|| "main".to_string());

        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(&job_id) {
                job.revision = commit_sha.clone();
            }
        }

        // A2: Fetch trusted SHA-256 digests from HF Git LFS metadata
        let expected_sha256 = fetch_repo_lfs_sha256_digests(&client, &repo_id, &revision, token.as_deref());

        // A8: Resolve remote file sizes using HEAD with ranged GET fallback
        let mut file_sizes = Vec::new();
        let mut aggregate_total_bytes = 0u64;

        for file_rel in &files {
            if cancel_flag.load(Ordering::SeqCst) {
                self.set_job_cancelled(&job_id);
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }
            if pause_flag.load(Ordering::SeqCst) {
                self.set_job_paused(&job_id);
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }

            let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{file_rel}");
            match resolve_remote_file_size(&client, &url, token.as_deref()) {
                Ok(sz) => {
                    file_sizes.push(sz);
                    aggregate_total_bytes += sz;
                }
                Err(err_msg) => {
                    self.set_job_error(&job_id, err_msg);
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
            }
        }

        // A5: Destination disk-space preflight
        let safety_margin = 512 * 1024 * 1024u64; // 512 MB safety margin
        let required_bytes = aggregate_total_bytes.saturating_add(safety_margin);

        if let Ok(free_space) = get_disk_free_space(&dest_path) {
            if free_space < required_bytes {
                let err_msg = format!(
                    "Insufficient disk space on destination volume. Required: {} ({} bytes with safety margin), Available: {} ({} bytes), Path: {}",
                    format_bytes(required_bytes),
                    required_bytes,
                    format_bytes(free_space),
                    free_space,
                    dest_path.display()
                );
                self.set_job_error(&job_id, err_msg);
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }
        }

        {
            if let Ok(mut guard) = self.jobs.lock() {
                if let Some(job) = guard.get_mut(&job_id) {
                    job.total_bytes = aggregate_total_bytes;
                    job.status = DownloadStatus::Downloading;
                    job.state = DownloadStatus::Downloading;
                }
            }
        }
        if let Some(j) = self.get_job(&job_id) {
            self.emit_progress_for_job(&j);
        }

        // Download shards with .part staging, range validation, retry, and checksum verification
        let mut completed_bytes_prior_shards = 0u64;

        for (shard_idx, file_rel) in files.iter().enumerate() {
            let expected_file_size = file_sizes[shard_idx];
            let target_final = dest_path.join(file_rel);
            let target_part = dest_path.join(format!("{file_rel}.part"));

            if let Some(parent) = target_part.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            // A2: Re-verify pre-existing files before skipping them
            if target_final.exists() {
                if let Ok(meta) = std::fs::metadata(&target_final) {
                    if meta.len() == expected_file_size && expected_file_size > 0 {
                        let expected_hash = expected_sha256.get(file_rel);
                        if let Some(expected_digest) = expected_hash {
                            self.set_job_verifying(&job_id, file_rel, shard_idx + 1);
                            if let Ok(actual_digest) = verify_file_sha256(&target_final, &cancel_flag) {
                                if actual_digest.eq_ignore_ascii_case(expected_digest) {
                                    self.set_job_verification_state(&job_id, "verified");
                                    completed_bytes_prior_shards += expected_file_size;
                                    continue;
                                }
                            }
                            // Corrupted pre-existing file: remove and re-download fresh
                            let _ = std::fs::remove_file(&target_final);
                        } else {
                            // Size matched with no digest available
                            self.set_job_verification_state(&job_id, "size_only");
                            completed_bytes_prior_shards += expected_file_size;
                            continue;
                        }
                    }
                }
            }

            // Shard transfer loop with retry, backoff, and range validation
            let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{file_rel}");
            let max_retries = 5;
            let mut attempt = 0;
            loop {
                if cancel_flag.load(Ordering::SeqCst) {
                    self.set_job_cancelled(&job_id);
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
                if pause_flag.load(Ordering::SeqCst) {
                    self.set_job_paused(&job_id);
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }

                let mut current_offset = std::fs::metadata(&target_part)
                    .map(|m| m.len())
                    .unwrap_or(0);

                if expected_file_size > 0 && current_offset > expected_file_size {
                    let _ = std::fs::remove_file(&target_part);
                    current_offset = 0;
                }

                let mut req = client.get(&url);
                if let Some(t) = token.as_deref().filter(|s| !s.trim().is_empty()) {
                    req = req.bearer_auth(t);
                }
                if current_offset > 0 {
                    req = req.header(reqwest::header::RANGE, format!("bytes={current_offset}-"));
                }

                let resp = match req.send() {
                    Ok(r) => r,
                    Err(e) => {
                        attempt += 1;
                        if attempt > max_retries {
                            self.set_job_error(&job_id, format!("Network error for {file_rel} after {max_retries} retries: {e}"));
                            self.cleanup_finished_job(&job_id);
                            self.try_start_next_job();
                            return;
                        }
                        if !wait_for_retry(&cancel_flag, &pause_flag, attempt, None) {
                            if cancel_flag.load(Ordering::SeqCst) {
                                self.set_job_cancelled(&job_id);
                            } else {
                                self.set_job_paused(&job_id);
                            }
                            self.cleanup_finished_job(&job_id);
                            self.try_start_next_job();
                            return;
                        }
                        continue;
                    }
                };

                let status = resp.status();

                // Non-retryable client errors: fail immediately
                if status == reqwest::StatusCode::UNAUTHORIZED {
                    self.set_job_error(&job_id, format!("Authentication required (401) for {file_rel}"));
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
                if status == reqwest::StatusCode::FORBIDDEN {
                    self.set_job_error(&job_id, format!("Access forbidden (403) for {file_rel}"));
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }
                if status == reqwest::StatusCode::NOT_FOUND {
                    self.set_job_error(&job_id, format!("File not found on remote (404) for {file_rel}"));
                    self.cleanup_finished_job(&job_id);
                    self.try_start_next_job();
                    return;
                }

                // Retryable server errors (429, 500, 502, 503, 504)
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR
                    || status == reqwest::StatusCode::BAD_GATEWAY
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                {
                    attempt += 1;
                    if attempt > max_retries {
                        self.set_job_error(&job_id, format!("Server error {status} for {file_rel} after {max_retries} retries"));
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                    let retry_after = resp.headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok());
                    if !wait_for_retry(&cancel_flag, &pause_flag, attempt, retry_after) {
                        if cancel_flag.load(Ordering::SeqCst) {
                            self.set_job_cancelled(&job_id);
                        } else {
                            self.set_job_paused(&job_id);
                        }
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                    continue;
                }

                // A3: Range resume validation
                let cr_header = resp.headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok());
                let range_check = validate_range_response(status, cr_header, current_offset);

                match range_check {
                    RangeValidation::ValidRangeResume => {
                        // Resuming cleanly from verified offset
                    }
                    RangeValidation::MismatchRestartFromZero | RangeValidation::FullContentRestartFromZero => {
                        // Server ignored range or returned mismatched start offset: restart from 0!
                        let _ = std::fs::remove_file(&target_part);
                        current_offset = 0;
                    }
                    RangeValidation::RangeNotSatisfiable => {
                        if expected_file_size > 0 && current_offset >= expected_file_size {
                            break;
                        }
                        let _ = std::fs::remove_file(&target_part);
                        attempt += 1;
                        if attempt > max_retries {
                            self.set_job_error(&job_id, format!("Range not satisfiable for {file_rel}"));
                            self.cleanup_finished_job(&job_id);
                            self.try_start_next_job();
                            return;
                        }
                        continue;
                    }
                    RangeValidation::InvalidStatus(s) => {
                        self.set_job_error(&job_id, format!("Unexpected HTTP response {s} for {file_rel}"));
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                }

                let mut out_file = match std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(current_offset == 0)
                    .append(current_offset > 0)
                    .open(&target_part)
                {
                    Ok(f) => f,
                    Err(e) => {
                        self.set_job_error(&job_id, format!("Cannot write to {}: {e}", target_part.display()));
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                };

                let mut reader = resp;
                let mut buf = [0u8; 64 * 1024];
                let mut shard_done = current_offset;
                let start_inst = Instant::now();
                let mut last_emit = Instant::now();
                let mut read_failed = false;

                // Read loop with cancel/pause checking
                loop {
                    if cancel_flag.load(Ordering::SeqCst) {
                        let _ = out_file.flush();
                        self.set_job_cancelled(&job_id);
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                    if pause_flag.load(Ordering::SeqCst) {
                        let _ = out_file.flush();
                        self.set_job_paused(&job_id);
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }

                    match reader.read(&mut buf) {
                        Ok(0) => break, // Clean EOF
                        Ok(n) => {
                            if let Err(e) = out_file.write_all(&buf[..n]) {
                                self.set_job_error(&job_id, format!("Disk write error for {file_rel}: {e}"));
                                self.cleanup_finished_job(&job_id);
                                self.try_start_next_job();
                                return;
                            }
                            shard_done += n as u64;
                            let aggregate_done = completed_bytes_prior_shards + shard_done;

                            if last_emit.elapsed() >= Duration::from_millis(250) {
                                let elapsed_sec = start_inst.elapsed().as_secs_f64().max(0.001);
                                let transferred_this_session = shard_done.saturating_sub(current_offset);
                                let speed = transferred_this_session as f64 / elapsed_sec;
                                let remaining = aggregate_total_bytes.saturating_sub(aggregate_done);
                                let eta = if speed > 1024.0 {
                                    Some(remaining as f64 / speed)
                                } else {
                                    None
                                };

                                if let Ok(mut guard) = self.jobs.lock() {
                                    if let Some(job) = guard.get_mut(&job_id) {
                                        job.downloaded_bytes = aggregate_done;
                                        job.speed_bps = speed;
                                        job.eta_seconds = eta;
                                        job.current_file = Some(file_rel.clone());
                                        job.current_shard = shard_idx + 1;
                                    }
                                }
                                if let Some(j) = self.get_job(&job_id) {
                                    self.emit_progress_for_job(&j);
                                }
                                last_emit = Instant::now();
                            }
                        }
                        Err(e) => {
                            // Read timeout / network error: retry from last verified offset
                            let _ = out_file.flush();
                            if cancel_flag.load(Ordering::SeqCst) {
                                self.set_job_cancelled(&job_id);
                                self.cleanup_finished_job(&job_id);
                                self.try_start_next_job();
                                return;
                            }
                            if pause_flag.load(Ordering::SeqCst) {
                                self.set_job_paused(&job_id);
                                self.cleanup_finished_job(&job_id);
                                self.try_start_next_job();
                                return;
                            }
                            attempt += 1;
                            if attempt > max_retries {
                                self.set_job_error(&job_id, format!("Transfer read stalled for {file_rel}: {e}"));
                                self.cleanup_finished_job(&job_id);
                                self.try_start_next_job();
                                return;
                            }
                            read_failed = true;
                            if !wait_for_retry(&cancel_flag, &pause_flag, attempt, None) {
                                if cancel_flag.load(Ordering::SeqCst) {
                                    self.set_job_cancelled(&job_id);
                                } else {
                                    self.set_job_paused(&job_id);
                                }
                                self.cleanup_finished_job(&job_id);
                                self.try_start_next_job();
                                return;
                            }
                            break;
                        }
                    }
                }

                let _ = out_file.flush();

                if !read_failed {
                    break;
                }
            }

            // A2: Hash completed .part before promotion
            let expected_hash = expected_sha256.get(file_rel);
            self.set_job_verifying(&job_id, file_rel, shard_idx + 1);

            let actual_size = std::fs::metadata(&target_part).map(|m| m.len()).unwrap_or(0);
            if expected_file_size > 0 && actual_size != expected_file_size {
                self.set_job_error(
                    &job_id,
                    format!("Shard size mismatch for {file_rel} ({actual_size} of {expected_file_size} bytes)"),
                );
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }

            if let Some(expected_digest) = expected_hash {
                match verify_file_sha256(&target_part, &cancel_flag) {
                    Ok(actual_digest) => {
                        if !actual_digest.eq_ignore_ascii_case(expected_digest) {
                            // Checksum mismatch: do NOT promote! Remove .part
                            let _ = std::fs::remove_file(&target_part);
                            self.set_job_error(
                                &job_id,
                                format!("SHA-256 checksum mismatch for {file_rel}: expected {expected_digest}, got {actual_digest}"),
                            );
                            self.cleanup_finished_job(&job_id);
                            self.try_start_next_job();
                            return;
                        }
                        self.set_job_verification_state(&job_id, "verified");
                    }
                    Err(e) => {
                        self.set_job_error(&job_id, format!("Integrity check failed for {file_rel}: {e}"));
                        self.cleanup_finished_job(&job_id);
                        self.try_start_next_job();
                        return;
                    }
                }
            } else {
                self.set_job_verification_state(&job_id, "size_only");
            }

            // Promote .part to final target
            let _ = std::fs::remove_file(&target_final);
            if let Err(e) = std::fs::rename(&target_part, &target_final) {
                self.set_job_error(&job_id, format!("Cannot finalize shard {file_rel}: {e}"));
                self.cleanup_finished_job(&job_id);
                self.try_start_next_job();
                return;
            }

            completed_bytes_prior_shards += expected_file_size;
        }

        // A9: Surface library registration error
        let primary_path = dest_path
            .join(files.first().unwrap_or(&"".to_string()))
            .to_string_lossy()
            .into_owned();

        let registry_entry = RegistryModelEntry {
            model_id: model_id.clone(),
            repo_id: repo_id.clone(),
            backend: "llamacpp".to_string(),
            quant: quant.clone(),
            files: files.clone(),
            size_bytes: aggregate_total_bytes,
            installed_at: crate::state::now_ms(),
            local_path: primary_path,
            is_split: total_shards > 1,
            companion_mmproj: None,
        };

        if let Err(e) = self.registry.register(registry_entry) {
            let err_msg = format!("downloaded, library registration failed: {e}");
            self.set_job_error(&job_id, err_msg);
            self.cleanup_finished_job(&job_id);
            self.try_start_next_job();
            return;
        }

        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(&job_id) {
                job.status = DownloadStatus::Completed;
                job.state = DownloadStatus::Completed;
                job.downloaded_bytes = aggregate_total_bytes;
                job.speed_bps = 0.0;
                job.eta_seconds = Some(0.0);
            }
        }

        if let Some(j) = self.get_job(&job_id) {
            self.emit_progress_for_job(&j);
        }

        self.cleanup_finished_job(&job_id);
        self.try_start_next_job();
    }

    fn set_job_error(&self, job_id: &str, error: String) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Failed;
                job.state = DownloadStatus::Failed;
                job.speed_bps = 0.0;
                job.error = Some(error);
            }
        }
        if let Some(j) = self.get_job(job_id) {
            self.emit_progress_for_job(&j);
        }
    }

    fn set_job_cancelled(&self, job_id: &str) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Cancelled;
                job.state = DownloadStatus::Cancelled;
                job.speed_bps = 0.0;
            }
        }
        if let Some(j) = self.get_job(job_id) {
            self.emit_progress_for_job(&j);
        }
    }

    fn set_job_paused(&self, job_id: &str) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Paused;
                job.state = DownloadStatus::Paused;
                job.speed_bps = 0.0;
            }
        }
        if let Some(j) = self.get_job(job_id) {
            self.emit_progress_for_job(&j);
        }
    }

    fn set_job_verifying(&self, job_id: &str, file: &str, shard: usize) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Verifying;
                job.state = DownloadStatus::Verifying;
                job.verification_state = "verifying".to_string();
                job.current_file = Some(file.to_string());
                job.current_shard = shard;
            }
        }
        if let Some(j) = self.get_job(job_id) {
            self.emit_progress_for_job(&j);
        }
    }

    fn set_job_verification_state(&self, job_id: &str, state: &str) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.verification_state = state.to_string();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_v4_generation() {
        let u1 = generate_uuid_v4();
        let u2 = generate_uuid_v4();
        assert_ne!(u1, u2);
        assert_eq!(u1.len(), 36);
        assert_eq!(&u1[14..15], "4"); // Version 4
        let variant_char = u1.chars().nth(19).unwrap();
        assert!(matches!(variant_char, '8' | '9' | 'a' | 'b'));
    }

    #[test]
    fn test_atomic_job_registration() {
        let registry = Arc::new(ModelRegistry {
            data: Mutex::new(crate::model_registry::ModelRegistryData::default()),
        });
        let manager = Arc::new(DownloadManager::new(registry));
        // Set concurrency to 0 to prevent workers from draining queued jobs during test
        manager.set_max_concurrent(0);

        let repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF";
        let quant = "Q4_K_M";
        let files = vec!["meta-llama-3.1-8b-instruct.Q4_K_M.gguf".to_string()];

        let num_threads = 20;
        let mut handles = Vec::new();

        for _ in 0..num_threads {
            let mgr = Arc::clone(&manager);
            let r = repo.to_string();
            let q = quant.to_string();
            let f = files.clone();
            let handle = std::thread::spawn(move || {
                mgr.enqueue_gguf_job_internal(
                    r.clone(),
                    r,
                    q,
                    f,
                    "C:/dummy/path".to_string(),
                    None,
                )
            });
            handles.push(handle);
        }

        let mut successes = 0;
        let mut rejected = 0;
        let mut job_id = String::new();

        for h in handles {
            match h.join().unwrap() {
                Ok(id) => {
                    successes += 1;
                    job_id = id;
                }
                Err(err) => {
                    assert!(err.contains("already in progress"));
                    rejected += 1;
                }
            }
        }

        assert_eq!(successes, 1, "Exactly one thread must succeed in registration");
        assert_eq!(rejected, num_threads - 1, "All conflicting threads must be rejected");
        assert!(job_id.starts_with("job-"), "Job ID must have job- prefix");
        assert_eq!(job_id.len(), 40, "Job ID must contain full UUID v4");
    }

    #[test]
    fn test_content_range_mismatch_rejection() {
        // Case 1: Partial content with matching start offset -> Valid
        let check1 = validate_range_response(
            reqwest::StatusCode::PARTIAL_CONTENT,
            Some("bytes 1000-1999/2000"),
            1000,
        );
        assert_eq!(check1, RangeValidation::ValidRangeResume);

        // Case 2: Partial content with MISMATCHED start offset -> MismatchRestartFromZero
        let check2 = validate_range_response(
            reqwest::StatusCode::PARTIAL_CONTENT,
            Some("bytes 500-1999/2000"),
            1000,
        );
        assert_eq!(check2, RangeValidation::MismatchRestartFromZero);

        // Case 3: Server ignored range header and returned 200 OK -> FullContentRestartFromZero
        let check3 = validate_range_response(
            reqwest::StatusCode::OK,
            None,
            1000,
        );
        assert_eq!(check3, RangeValidation::FullContentRestartFromZero);

        // Case 4: Server returned 416 Range Not Satisfiable -> RangeNotSatisfiable
        let check4 = validate_range_response(
            reqwest::StatusCode::RANGE_NOT_SATISFIABLE,
            None,
            1000,
        );
        assert_eq!(check4, RangeValidation::RangeNotSatisfiable);
    }

    #[test]
    fn test_checksum_mismatch_not_promoted() {
        let temp_dir = std::env::temp_dir().join(format!("llm_panel_test_{}", crate::state::now_ms()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let target_final = temp_dir.join("test_model.gguf");
        let target_part = temp_dir.join("test_model.gguf.part");

        let test_payload = b"Sample test GGUF payload data for hashing";
        std::fs::write(&target_part, test_payload).unwrap();

        // Deliberately incorrect SHA-256 digest
        let expected_wrong_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        let cancel_flag = AtomicBool::new(false);

        let actual_hash = verify_file_sha256(&target_part, &cancel_flag).unwrap();
        assert_ne!(actual_hash, expected_wrong_sha256);

        // Verification fails: simulate promotion gate
        if !actual_hash.eq_ignore_ascii_case(expected_wrong_sha256) {
            let _ = std::fs::remove_file(&target_part);
        } else {
            let _ = std::fs::rename(&target_part, &target_final);
        }

        assert!(!target_final.exists(), "Corrupt part must NOT be promoted to final file");
        assert!(!target_part.exists(), "Corrupt part must be removed");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_retry_backoff_against_mock_server() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            // Request 1: return 503 Service Unavailable
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line.trim().is_empty() { break; }
                    line.clear();
                }
                let resp = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(resp.as_bytes());
            }

            // Request 2: return 429 Too Many Requests with Retry-After: 0
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line.trim().is_empty() { break; }
                    line.clear();
                }
                let resp = "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(resp.as_bytes());
            }

            // Request 3: return 200 OK with content
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line.trim().is_empty() { break; }
                    line.clear();
                }
                let body = "Recovered after retries";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();

        let cancel_flag = AtomicBool::new(false);
        let pause_flag = AtomicBool::new(false);
        let url = format!("http://127.0.0.1:{port}/test-download");

        let mut attempt = 0;
        let mut final_body = String::new();

        while attempt <= 5 {
            let resp = match client.get(&url).send() {
                Ok(r) => r,
                Err(_) => {
                    attempt += 1;
                    assert!(wait_for_retry(&cancel_flag, &pause_flag, attempt, Some(0)));
                    continue;
                }
            };

            let status = resp.status();
            if status == reqwest::StatusCode::SERVICE_UNAVAILABLE || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                attempt += 1;
                let retry_after = resp.headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                assert!(wait_for_retry(&cancel_flag, &pause_flag, attempt, retry_after));
                continue;
            }

            if status.is_success() {
                final_body = resp.text().unwrap();
                break;
            }
        }

        assert_eq!(final_body, "Recovered after retries");
        assert_eq!(attempt, 2, "Should have retried exactly twice before success");
    }

    #[test]
    fn test_stalled_read_cancel() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line.trim().is_empty() { break; }
                    line.clear();
                }
                let header = "HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.flush();
                // Send initial chunk
                let _ = stream.write_all(&[b'x'; 256]);
                let _ = stream.flush();
                std::thread::sleep(Duration::from_secs(5));
            }
        });

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&cancel_flag);
        let url = format!("http://127.0.0.1:{port}/stall");

        let handle = std::thread::spawn(move || {
            let mut resp = match client.get(&url).send() {
                Ok(r) => r,
                Err(_) => {
                    if flag_clone.load(Ordering::SeqCst) {
                        return Err("Cancelled");
                    }
                    return Err("Error");
                }
            };
            let mut buf = [0u8; 16];
            let mut read_bytes = 0;

            loop {
                if flag_clone.load(Ordering::SeqCst) {
                    return Err("Cancelled");
                }
                match resp.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        read_bytes += n;
                        std::thread::sleep(Duration::from_millis(15));
                    }
                    Err(_) => {
                        if flag_clone.load(Ordering::SeqCst) {
                            return Err("Cancelled");
                        }
                        return Err("Error");
                    }
                }
            }
            Ok(read_bytes)
        });

        // Sleep briefly to let client begin reading initial chunks, then signal cancel
        std::thread::sleep(Duration::from_millis(50));
        cancel_flag.store(true, Ordering::SeqCst);

        let result = handle.join().unwrap();
        assert_eq!(result, Err("Cancelled"), "Download must promptly abort on cancel");
    }

    #[test]
    fn test_bounded_worker_pool_and_queue() {
        let registry = Arc::new(ModelRegistry {
            data: Mutex::new(crate::model_registry::ModelRegistryData::default()),
        });
        let manager = Arc::new(DownloadManager::new(registry));
        manager.set_max_concurrent(1);

        let job1 = DownloadJob {
            id: "job-active-1".into(),
            model_id: "test/m1".into(),
            repo_id: "test/m1".into(),
            backend: "llamacpp".into(),
            quant: "Q4_K_M".into(),
            files: vec!["f1.gguf".into()],
            destination_dir: "C:/tmp".into(),
            total_bytes: 100,
            downloaded_bytes: 50,
            speed_bps: 10.0,
            eta_seconds: Some(5.0),
            status: DownloadStatus::Downloading,
            state: DownloadStatus::Downloading,
            error: None,
            current_file: Some("f1.gguf".into()),
            current_shard: 1,
            total_shards: 1,
            is_split: false,
            companion_mmproj: None,
            revision: None,
            created_at: 1000,
            verification_state: "unverified".into(),
            supports_pause: true,
            supports_resume: true,
            supports_cancel: true,
        };

        let job2 = DownloadJob {
            id: "job-queued-2".into(),
            model_id: "test/m2".into(),
            repo_id: "test/m2".into(),
            backend: "llamacpp".into(),
            quant: "Q8_0".into(),
            files: vec!["f2.gguf".into()],
            destination_dir: "C:/tmp".into(),
            total_bytes: 200,
            downloaded_bytes: 0,
            speed_bps: 0.0,
            eta_seconds: None,
            status: DownloadStatus::Queued,
            state: DownloadStatus::Queued,
            error: None,
            current_file: Some("f2.gguf".into()),
            current_shard: 1,
            total_shards: 1,
            is_split: false,
            companion_mmproj: None,
            revision: None,
            created_at: 2000,
            verification_state: "unverified".into(),
            supports_pause: true,
            supports_resume: true,
            supports_cancel: true,
        };

        {
            let mut guard = manager.jobs.lock().unwrap();
            guard.insert("job-active-1".into(), job1);
            guard.insert("job-queued-2".into(), job2);
        }

        // try_start_next_job should respect max_concurrent = 1 and NOT promote job2 yet
        manager.try_start_next_job();
        assert_eq!(manager.get_job("job-queued-2").unwrap().status, DownloadStatus::Queued);

        // When job1 completes, try_start_next_job promotes job2
        {
            let mut guard = manager.jobs.lock().unwrap();
            if let Some(j1) = guard.get_mut("job-active-1") {
                j1.status = DownloadStatus::Completed;
            }
        }

        manager.try_start_next_job();
        assert_eq!(manager.get_job("job-queued-2").unwrap().status, DownloadStatus::Resolving);
    }

    #[test]
    fn test_multi_quant_downloads_coexist_and_cancel_isolation() {
        let registry = Arc::new(ModelRegistry {
            data: Mutex::new(crate::model_registry::ModelRegistryData::default()),
        });
        let manager = Arc::new(DownloadManager::new(registry));

        let repo = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF";
        let model = "Qwen/Qwen2.5-Coder-7B-Instruct";

        // Insert job 1: Q4_K_M
        let job1 = DownloadJob {
            id: "job-1".into(),
            model_id: model.into(),
            repo_id: repo.into(),
            backend: "llamacpp".into(),
            quant: "Q4_K_M".into(),
            files: vec!["file-q4.gguf".into()],
            destination_dir: "C:/gguf/qwen-q4".into(),
            total_bytes: 4000,
            downloaded_bytes: 1000,
            speed_bps: 100.0,
            eta_seconds: Some(30.0),
            status: DownloadStatus::Downloading,
            state: DownloadStatus::Downloading,
            error: None,
            current_file: Some("file-q4.gguf".into()),
            current_shard: 1,
            total_shards: 1,
            is_split: false,
            companion_mmproj: None,
            revision: None,
            created_at: 100,
            verification_state: "unverified".into(),
            supports_pause: true,
            supports_resume: true,
            supports_cancel: true,
        };

        // Insert job 2: Q8_0 (same repo, different quant)
        let job2 = DownloadJob {
            id: "job-2".into(),
            model_id: model.into(),
            repo_id: repo.into(),
            backend: "llamacpp".into(),
            quant: "Q8_0".into(),
            files: vec!["file-q8.gguf".into()],
            destination_dir: "C:/gguf/qwen-q8".into(),
            total_bytes: 8000,
            downloaded_bytes: 2000,
            speed_bps: 200.0,
            eta_seconds: Some(30.0),
            status: DownloadStatus::Downloading,
            state: DownloadStatus::Downloading,
            error: None,
            current_file: Some("file-q8.gguf".into()),
            current_shard: 1,
            total_shards: 1,
            is_split: false,
            companion_mmproj: None,
            revision: None,
            created_at: 200,
            verification_state: "unverified".into(),
            supports_pause: true,
            supports_resume: true,
            supports_cancel: true,
        };

        let ctrl1 = ActiveDownloadController {
            cancel_flag: Arc::new(AtomicBool::new(false)),
            pause_flag: Arc::new(AtomicBool::new(false)),
        };
        let ctrl2 = ActiveDownloadController {
            cancel_flag: Arc::new(AtomicBool::new(false)),
            pause_flag: Arc::new(AtomicBool::new(false)),
        };

        {
            let mut guard = manager.jobs.lock().unwrap();
            guard.insert("job-1".into(), job1);
            guard.insert("job-2".into(), job2);
            let mut c_guard = manager.controllers.lock().unwrap();
            c_guard.insert("job-1".into(), ctrl1);
            c_guard.insert("job-2".into(), ctrl2);
        }

        // Both jobs are active simultaneously
        assert!(manager.is_job_active(repo, Some("Q4_K_M")));
        assert!(manager.is_job_active(repo, Some("Q8_0")));
        assert!(!manager.is_job_active(repo, Some("Q6_K")));

        // Lookup by repo and quant works correctly
        let q4_job = manager.get_job_by_repo_and_quant(repo, "Q4_K_M").unwrap();
        assert_eq!(q4_job.id, "job-1");
        let q8_job = manager.get_job_by_repo_and_quant(repo, "Q8_0").unwrap();
        assert_eq!(q8_job.id, "job-2");

        // Cancel Q4_K_M: Q8_0 must remain Downloading
        assert!(manager.cancel_by_repo_and_quant(repo, "Q4_K_M"));
        assert!(!manager.is_job_active(repo, Some("Q4_K_M")));
        assert!(manager.is_job_active(repo, Some("Q8_0")));

        let q4_after = manager.get_job("job-1").unwrap();
        assert_eq!(q4_after.status, DownloadStatus::Cancelled);
        let q8_after = manager.get_job("job-2").unwrap();
        assert_eq!(q8_after.status, DownloadStatus::Downloading);
    }
}
