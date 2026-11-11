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
