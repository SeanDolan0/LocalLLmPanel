# Local LLM Panel

Desktop application managing local LLM inference engines (vLLM via WSL2 and native llama.cpp on Windows).

## Language

**Server Definition**:
The persisted configuration and runtime options required to launch a local model instance.
_Avoid_: server config, config row, recipe

**Live Server**:
An active, supervised operating system process executing a local inference model.
_Avoid_: running server, instance, worker process

**Server Manager**:
The central authority governing server lifecycle, port allocation, and hardware resource reservations.
_Avoid_: server supervisor, controller, server service

**Backend Driver**:
An execution adapter that isolates OS-specific process mechanics between Linux inside WSL2 and native Windows.
_Avoid_: process runner, launcher, executor

**Server Snapshot**:
A point-in-time summary of a server's operational health, process liveliness, and runtime metrics.
_Avoid_: server status, server row, server state

**VRAM Reservation**:
An exclusive GPU memory allotment tracked during server startup to prevent concurrent memory oversubscription.
_Avoid_: VRAM lock, memory guard, GPU allocation

**Hardware Sizer**:
The authoritative deep module managing hardware detection, memory topology, dynamic telemetry, VRAM reservations, and model fit scoring.
_Avoid_: hardware detector, memory estimator, fit calculator

**Hardware Snapshot**:
A point-in-time view of host and WSL2 compute environments including CPU topology, GPUs, physical and effective free VRAM, and distinct host/WSL RAM pools.
_Avoid_: system specs, gpu info, hardware profile

**Allocation Report**:
Precomputed resource pressure, active weight footprints, system RAM spill, and memory alert states calculated respecting user MemorySettings.
_Avoid_: memory stats, vram breakdown, spill calc

**Fit Report**:
The authoritative verdict evaluating whether a model variant fits in memory, its execution mode (GPU, Hybrid Offload, CPU), estimated context capacity, and expected throughput.
_Avoid_: score verdict, fit result, recommendation score

**Model Catalog**:
The authoritative deep module managing online model discovery, HuggingFace metadata resolution, local artifact inventory reconciliation, chat template extraction, and draft model compatibility.
_Avoid_: model service, discovery manager, model indexer

**Catalog Model**:
A normalized model candidate with publisher tier, family taxonomy, parameter scale, available quant formats, and sizing scores.
_Avoid_: model DTO, discovery item, feed row

**Catalog Model Details**:
Comprehensive model card metadata including architectural context limits, layer/head geometry, verified quantization variants with fit verdicts, and natural language explanations.
_Avoid_: model info, model card data, stats payload

**Installed Model**:
A verified runnable model present on local disk (WSL Hugging Face hub cache, imported folder, or native GGUF directory).
_Avoid_: local model, downloaded model, library item

**Draft Model Association**:
A catalog relationship mapping a primary target model to compatible low-latency draft models sharing architecture and tokenizer vocabulary for speculative decoding.
_Avoid_: speculative pair, draft config, assistant model
