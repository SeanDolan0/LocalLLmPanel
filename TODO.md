# TODO

## Instructions for AI
Work through the tasks under **To Do** from top to bottom.
- Do one task at a time, fully, before starting the next.
- When you finish a task, change `[ ]` to `[x]` and move it to **Done**, adding a one-line note on what you changed.
- If a task is unclear or blocked, don't guess: move it to **Blocked** with a short reason and continue with the next task.
- Don't do anything that isn't on this list. If you spot something worth doing, add it to **Suggestions** instead.
- Stop when To Do is empty and give me a short summary.

## To Do


## Blocked
<!-- AI: move stuck tasks here with the reason -->

## Done
- [x] Task 4: tqdm progress frames are '\r'-delimited (one trailing '\n'), so BufReader::lines() collapsed the whole bar into one line. run_script_stream now splits on '\r' too; parse_progress_line moved to module scope for testability (regression test uses real hf frames). Download bar now receives per-frame percentages.
- [x] Task 1: Removed the "Auto-Resume Running Servers" feature entirely (config field, Settings toggle, resume_servers_if_configured, startup spawn). Servers now only start manually.
- [x] Task 2: servers_stop/delete/restart now run stop_server off the main thread via spawn_blocking (was: sync Tauri command blocking the UI during the up-to-30s process unload).
- [x] Task 3: GGUF pulls now route through the native download_gguf command → Windows %APPDATA%\local-llm-panel\gguf (WSL hf download only used for vLLM models). Added ggufFiles/downloadGguf api methods and GGUF repo detection in Search.tsx pull().

## Suggestions
<!-- AI: ideas you noticed but didn't act on -->

## Notes / Context
<!-- Optional: anything the AI should know (file locations, constraints, preferences) -->