//! Local LLM Panel — Tauri 2 app managing vLLM inside WSL2.

pub mod commands;
pub mod estimate;
pub mod fit;
pub mod gateway;
pub mod hf;
pub mod llmfit_adapter;
pub mod llamacpp_install;
pub mod provision;
pub mod security;
pub mod server;
pub mod state;
pub mod wsl;

use state::AppState;
use std::sync::Arc;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::Manager;

#[cfg(test)]
mod it;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(AppState::new()))
        .setup(|app| {
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let show_i = MenuItem::with_id(app, "show", "Open Panel", true, None::<&str>)?;
            let stop_all_i =
                MenuItem::with_id(app, "stop_all", "Stop All Servers", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &stop_all_i, &quit_i])?;

            if let Some(icon) = app.default_window_icon().cloned() {
                let _tray = TrayIconBuilder::new()
                    .icon(icon)
                    .menu(&menu)
                    .show_menu_on_left_click(false)
                    .on_menu_event(|app, event| match event.id.as_ref() {
                        "quit" => {
                            app.exit(0);
                        }
                        "show" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "stop_all" => {
                            let app_handle = app.clone();
                            tauri::async_runtime::spawn(async move {
                                let state: tauri::State<Arc<AppState>> = app_handle.state();
                                let server_ids: Vec<String> = {
                                    let srvs = state.servers.lock().unwrap();
                                    srvs.keys().cloned().collect()
                                };
                                for id in server_ids {
                                    let st = Arc::clone(&state);
                                    let handle = app_handle.clone();
                                    let _ = tokio::task::spawn_blocking(move || {
                                        crate::server::stop_server(&st, Some(&handle), &id)
                                    })
                                    .await;
                                }
                            });
                        }
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } = event
                        {
                            let app = tray.app_handle();
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    })
                    .build(app)?;
            }

            if let Some(window) = app.get_webview_window("main") {
                let window_clone = window.clone();
                let app_handle = app.handle().clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        let state: tauri::State<Arc<AppState>> = app_handle.state();
                        let minimize = state.config.lock().unwrap().minimize_to_tray;
                        if minimize {
                            api.prevent_close();
                            let _ = window_clone.hide();
                        }
                    }
                });
            }

            let app_handle_for_resume = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let state: tauri::State<Arc<AppState>> = app_handle_for_resume.state();
                server::resume_servers_if_configured(&state, Some(&app_handle_for_resume)).await;
            });

            gateway::spawn_supervisor(app.handle().clone());

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::env_status,
            commands::provision,
            commands::install_llamacpp,
            commands::search_models,
            commands::search_models_with_fit,
            commands::recommended_models,
            commands::model_stats,
            commands::pull_model,
            commands::pull_status,
            commands::pull_cancel,
            commands::servers_list,
            commands::servers_create,
            commands::servers_delete,
            commands::servers_start,
            commands::servers_stop,
            commands::servers_restart,
            commands::servers_logs,
            commands::servers_metrics,
            commands::servers_chat,
            commands::servers_test_tool_call,
            commands::servers_chat_stream,
            commands::servers_chat_cancel,
            commands::conversations_list,
            commands::conversations_save,
            commands::conversations_delete,
            commands::benchmarks_run,
            commands::benchmarks_cancel,
            commands::benchmarks_history,
            commands::library_list,
            commands::gguf_files,
            commands::download_gguf,
            commands::library_remove,
            commands::library_disk_usage,
            commands::library_import_local,
            commands::settings_get,
            commands::settings_set,
            commands::gateway_status,
            commands::wslconfig_get,
            commands::gpu_status,
            commands::system_metrics_series,
            commands::server_metrics_series,
            commands::get_memory_settings,
            commands::update_memory_settings,
            commands::get_system_memory,
            commands::open_url,
            commands::wsl_distros,
            commands::autostart_get,
            commands::autostart_set,
            commands::config_export,
            commands::config_import,
            commands::server_recipe_export,
            commands::server_recipe_parse,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
