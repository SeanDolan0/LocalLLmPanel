//! Local LLM Panel — Tauri 2 app managing vLLM inside WSL2.

pub mod commands;
pub mod estimate;
pub mod fit;
pub mod hf;
pub mod provision;
pub mod server;
pub mod state;
pub mod wsl;

use state::AppState;

#[cfg(test)]
mod it;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(std::sync::Arc::new(AppState::new()))
        .invoke_handler(tauri::generate_handler![
            commands::env_status,
            commands::provision,
            commands::search_models,
            commands::search_models_with_fit,
            commands::recommended_models,
            commands::model_stats,
            commands::pull_model,
            commands::pull_status,
            commands::servers_list,
            commands::servers_create,
            commands::servers_delete,
            commands::servers_start,
            commands::servers_stop,
            commands::servers_restart,
            commands::servers_logs,
            commands::servers_metrics,
            commands::servers_chat,
            commands::library_list,
            commands::settings_get,
            commands::settings_set,
            commands::wslconfig_get,
            commands::gpu_status,
            commands::get_memory_settings,
            commands::update_memory_settings,
            commands::get_system_memory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}