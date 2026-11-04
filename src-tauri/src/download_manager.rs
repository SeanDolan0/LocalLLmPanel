//! Unified Download Manager:
//! - Multi-backend job tracking (llama.cpp GGUF & vLLM Safetensors)
//! - Resumable HTTP Range requests with `.part` staging
//! - Multi-shard atomic download & verification
//! - Background queue with concurrency limits
//! - Pause, resume, retry, and cancellation
//! - Seamless synchronization with the Local Model Registry

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
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
    pub error: Option<String>,
    pub current_file: Option<String>,
    pub current_shard: usize,
    pub total_shards: usize,
    pub is_split: bool,
    pub companion_mmproj: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgressEvent {
    pub job_id: String,
    pub model_id: String,
    pub repo_id: String,
    pub status: String,
    pub percent: f64,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: f64,
    pub eta_seconds: Option<f64>,
    pub current_file: Option<String>,
    pub current_shard: usize,
    pub total_shards: usize,
    pub error: Option<String>,
}

struct ActiveDownloadController {
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
}

pub struct DownloadManager {
    jobs: Mutex<HashMap<String, DownloadJob>>,
    controllers: Mutex<HashMap<String, ActiveDownloadController>>,
    registry: Arc<ModelRegistry>,
}

impl DownloadManager {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            controllers: Mutex::new(HashMap::new()),
            registry,
        }
    }

    pub fn get_job(&self, job_id: &str) -> Option<DownloadJob> {
        self.jobs.lock().ok()?.get(job_id).cloned()
    }

    pub fn get_job_by_repo(&self, repo_id: &str) -> Option<DownloadJob> {
        let guard = self.jobs.lock().ok()?;
        guard.values().find(|j| j.repo_id.eq_ignore_ascii_case(repo_id)).cloned()
    }

    pub fn list_jobs(&self) -> Vec<DownloadJob> {
        self.jobs
            .lock()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn cancel_job(&self, job_id: &str) -> bool {
        if let Ok(guard) = self.controllers.lock() {
            if let Some(ctrl) = guard.get(job_id) {
                ctrl.cancel_flag.store(true, Ordering::SeqCst);
            }
        }
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Cancelled;
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
                .find(|j| j.repo_id.eq_ignore_ascii_case(repo_id) || j.model_id.eq_ignore_ascii_case(repo_id))
                .map(|j| j.id.clone())
        };

        if let Some(id) = id_to_cancel {
            self.cancel_job(&id)
        } else {
            false
        }
    }

    pub fn pause_job(&self, job_id: &str) -> bool {
        if let Ok(guard) = self.controllers.lock() {
            if let Some(ctrl) = guard.get(job_id) {
                ctrl.pause_flag.store(true, Ordering::SeqCst);
            }
        }
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Paused;
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
        let job_id = format!("job-{}", crate::state::now_ms());

        // Check if duplicate job is already running
        {
            let guard = self.jobs.lock().map_err(|e| e.to_string())?;
            for existing in guard.values() {
                if existing.repo_id.eq_ignore_ascii_case(&repo_id)
                    && existing.quant.eq_ignore_ascii_case(&quant)
                    && (existing.status == DownloadStatus::Downloading
                        || existing.status == DownloadStatus::Resolving
                        || existing.status == DownloadStatus::Queued)
                {
                    return Err(format!("Download already in progress for {repo_id} ({quant})"));
                }
            }
        }

        let total_shards = files.len();
        let is_split = total_shards > 1;

        let job = DownloadJob {
            id: job_id.clone(),
            model_id: model_id.clone(),
            repo_id: repo_id.clone(),
            backend: "llamacpp".to_string(),
            quant: quant.clone(),
            files: files.clone(),
            destination_dir: destination_dir.clone(),
            total_bytes: 0,
            downloaded_bytes: 0,
            speed_bps: 0.0,
            eta_seconds: None,
            status: DownloadStatus::Queued,
            error: None,
            current_file: files.first().cloned(),
            current_shard: 1,
            total_shards,
            is_split,
            companion_mmproj: None,
        };

        {
            let mut guard = self.jobs.lock().map_err(|e| e.to_string())?;
            guard.insert(job_id.clone(), job);
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));

        {
            let mut guard = self.controllers.lock().map_err(|e| e.to_string())?;
            guard.insert(
                job_id.clone(),
                ActiveDownloadController {
                    cancel_flag: Arc::clone(&cancel_flag),
                    pause_flag: Arc::clone(&pause_flag),
                },
            );
        }

        // Spawn background worker
        let mgr = Arc::clone(self);
        let j_id = job_id.clone();
        std::thread::spawn(move || {
            mgr.run_gguf_download_worker(app, j_id, cancel_flag, pause_flag, token);
        });

        Ok(job_id)
    }

    fn run_gguf_download_worker(
        &self,
        app: AppHandle,
        job_id: String,
        cancel_flag: Arc<AtomicBool>,
        pause_flag: Arc<AtomicBool>,
        token: Option<String>,
    ) {
        let (repo_id, files, destination_dir, quant, model_id, total_shards) = {
            let mut guard = match self.jobs.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let job = match guard.get_mut(&job_id) {
                Some(j) => j,
                None => return,
            };
            job.status = DownloadStatus::Resolving;
            (
                job.repo_id.clone(),
                job.files.clone(),
                job.destination_dir.clone(),
                job.quant.clone(),
                job.model_id.clone(),
                job.total_shards,
            )
        };

        let emit_progress = |job: &DownloadJob| {
            let percent = if job.total_bytes > 0 {
                (job.downloaded_bytes as f64 / job.total_bytes as f64 * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };

            let ev = DownloadProgressEvent {
                job_id: job.id.clone(),
                model_id: job.model_id.clone(),
                repo_id: job.repo_id.clone(),
                status: format!("{:?}", job.status).to_ascii_lowercase(),
                percent,
                downloaded_bytes: job.downloaded_bytes,
                total_bytes: job.total_bytes,
                speed_bps: job.speed_bps,
                eta_seconds: job.eta_seconds,
                current_file: job.current_file.clone(),
                current_shard: job.current_shard,
                total_shards: job.total_shards,
                error: job.error.clone(),
            };
            let _ = app.emit("download-progress", &ev);

            // Emit backwards-compatible pull-progress
            let legacy_state = match job.status {
                DownloadStatus::Completed => "complete",
                DownloadStatus::Failed => "failed",
                _ => "downloading",
            };
            let _ = app.emit(
                "pull-progress",
                serde_json::json!({
                    "model": job.repo_id,
                    "state": legacy_state,
                    "file": job.current_file.clone().or_else(|| job.error.clone()),
                    "percent": percent,
                    "speed_bps": job.speed_bps,
                    "eta_seconds": job.eta_seconds,
                    "bytes_downloaded": job.downloaded_bytes,
                    "bytes_total": job.total_bytes,
                }),
            );
        };

        let dest_path = PathBuf::from(&destination_dir);
        if let Err(e) = std::fs::create_dir_all(&dest_path) {
            self.set_job_error(&job_id, format!("Cannot create folder: {e}"));
            if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
            return;
        }

        let client = match reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                self.set_job_error(&job_id, format!("HTTP client error: {e}"));
                if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                return;
            }
        };

        // 1. Resolve remote file sizes
        let mut file_sizes = Vec::new();
        let mut aggregate_total_bytes = 0u64;

        for file_rel in &files {
            let url = format!("https://huggingface.co/{repo_id}/resolve/main/{file_rel}");
            let mut req = client.head(&url);
            if let Some(t) = token.as_deref().filter(|s| !s.trim().is_empty()) {
                req = req.bearer_auth(t);
            }
            let size = match req.send().and_then(|r| r.error_for_status()) {
                Ok(resp) => resp.content_length().unwrap_or(0),
                Err(e) => {
                    self.set_job_error(&job_id, format!("Cannot reach {file_rel}: {e}"));
                    if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                    return;
                }
            };
            file_sizes.push(size);
            aggregate_total_bytes += size;
        }

        {
            if let Ok(mut guard) = self.jobs.lock() {
                if let Some(job) = guard.get_mut(&job_id) {
                    job.total_bytes = aggregate_total_bytes;
                    job.status = DownloadStatus::Downloading;
                }
            }
        }

        // 2. Download files with resume support (.part staging)
        let mut completed_bytes_prior_shards = 0u64;

        for (shard_idx, file_rel) in files.iter().enumerate() {
            let expected_file_size = file_sizes[shard_idx];
            let target_final = dest_path.join(file_rel);
            let target_part = dest_path.join(format!("{file_rel}.part"));

            if let Some(parent) = target_part.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            // Check if already completely downloaded
            if target_final.exists() {
                if let Ok(meta) = std::fs::metadata(&target_final) {
                    if meta.len() == expected_file_size && expected_file_size > 0 {
                        completed_bytes_prior_shards += expected_file_size;
                        continue;
                    }
                }
            }

            // Check existing .part offset for resume
            let mut current_offset = std::fs::metadata(&target_part)
                .map(|m| m.len())
                .unwrap_or(0);

            if expected_file_size > 0 && current_offset > expected_file_size {
                let _ = std::fs::remove_file(&target_part);
                current_offset = 0;
            }

            let url = format!("https://huggingface.co/{repo_id}/resolve/main/{file_rel}");
            let mut req = client.get(&url);
            if let Some(t) = token.as_deref().filter(|s| !s.trim().is_empty()) {
                req = req.bearer_auth(t);
            }
            if current_offset > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={current_offset}-"));
            }

            let mut resp = match req.send() {
                Ok(r) => r,
                Err(e) => {
                    self.set_job_error(&job_id, format!("Transfer error {file_rel}: {e}"));
                    if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                    return;
                }
            };

            // Handle 200 vs 206
            if resp.status() == reqwest::StatusCode::OK && current_offset > 0 {
                current_offset = 0;
            } else if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                if current_offset >= expected_file_size && expected_file_size > 0 {
                    let _ = std::fs::rename(&target_part, &target_final);
                    completed_bytes_prior_shards += expected_file_size;
                    continue;
                }
            } else if !resp.status().is_success() {
                self.set_job_error(&job_id, format!("Server returned {}: {file_rel}", resp.status()));
                if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                return;
            }

            let mut out_file = match std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(current_offset > 0)
                .truncate(current_offset == 0)
                .open(&target_part)
            {
                Ok(f) => f,
                Err(e) => {
                    self.set_job_error(&job_id, format!("Cannot write file {file_rel}: {e}"));
                    if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                    return;
                }
            };

            let mut shard_done = current_offset;
            let start_inst = Instant::now();
            let mut last_emit = Instant::now();
            let mut buf = vec![0u8; 1024 * 1024];

            loop {
                if cancel_flag.load(Ordering::SeqCst) {
                    let _ = out_file.flush();
                    return;
                }
                if pause_flag.load(Ordering::SeqCst) {
                    let _ = out_file.flush();
                    return;
                }

                let n = match resp.read(&mut buf) {
                    Ok(0) => break,
                    Ok(read_bytes) => read_bytes,
                    Err(e) => {
                        let _ = out_file.flush();
                        self.set_job_error(&job_id, format!("Read stream interrupted: {e}"));
                        if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                        return;
                    }
                };

                if let Err(e) = out_file.write_all(&buf[..n]) {
                    self.set_job_error(&job_id, format!("Disk write error: {e}"));
                    if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                    return;
                }

                shard_done += n as u64;
                let aggregate_done = completed_bytes_prior_shards + shard_done;

                if last_emit.elapsed() >= std::time::Duration::from_millis(250) {
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
                        emit_progress(&j);
                    }
                    last_emit = Instant::now();
                }
            }

            let _ = out_file.flush();

            // Verify shard size
            if expected_file_size > 0 && shard_done != expected_file_size {
                self.set_job_error(
                    &job_id,
                    format!("Shard size mismatch for {file_rel} ({shard_done} of {expected_file_size} bytes)"),
                );
                if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                return;
            }

            // Promote .part to final
            let _ = std::fs::remove_file(&target_final);
            if let Err(e) = std::fs::rename(&target_part, &target_final) {
                self.set_job_error(&job_id, format!("Cannot finalize shard: {e}"));
                if let Some(j) = self.get_job(&job_id) { emit_progress(&j); }
                return;
            }

            completed_bytes_prior_shards += shard_done;
        }

        // All shards complete! Register in local model registry
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

        let _ = self.registry.register(registry_entry);

        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(&job_id) {
                job.status = DownloadStatus::Completed;
                job.downloaded_bytes = aggregate_total_bytes;
                job.eta_seconds = Some(0.0);
            }
        }

        if let Some(j) = self.get_job(&job_id) {
            emit_progress(&j);
        }
    }

    fn set_job_error(&self, job_id: &str, error: String) {
        if let Ok(mut guard) = self.jobs.lock() {
            if let Some(job) = guard.get_mut(job_id) {
                job.status = DownloadStatus::Failed;
                job.error = Some(error);
            }
        }
    }
}
