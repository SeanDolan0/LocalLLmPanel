# About Local LLM Panel

Local LLM Panel is a Windows desktop application for managing local language-model inference without stitching together separate tools for model discovery, downloads, server startup, API routing, and performance monitoring.

It is built for people using their own NVIDIA-powered Windows PCs who want a practical control surface for local models. The goal is to make model selection and server operation easier to understand while leaving the underlying inference engines in control of inference.

## What the project brings together

- **Hardware-aware discovery:** model metadata, quantization variants, VRAM/RAM fit estimates, context planning, and estimated generation speed.
- **Two local inference backends:** vLLM running in WSL2 and native Windows llama.cpp.
- **Model lifecycle management:** Hugging Face downloads, GGUF downloads, local library scanning, and safe removal of unused model files.
- **A shared API surface:** an optional OpenAI-compatible gateway that routes requests to running backends.
- **Operational visibility:** process logs, health checks, GPU/server metrics, and repeatable benchmarks.

## Architecture

The desktop application uses:

- **Tauri 2 and Rust** for native commands, process lifecycle, configuration, hardware integration, server orchestration, and the API gateway.
- **React, TypeScript, and Vite** for the user interface.
- **WSL2 + Ubuntu + vLLM** for the vLLM backend.
- **Native Windows llama.cpp** for GGUF inference.
- **Hugging Face and GitHub APIs** for model metadata and release discovery.

Model search and downloads require internet access. Inference is performed locally by the selected backend; the application is not a hosted inference service and does not provide bundled model weights.

## Design principles

1. **Measured performance over false precision.** Hardware-fit and throughput figures are estimates, and the UI should distinguish estimates from measured benchmark results.
2. **Make resource constraints visible.** Model weights, KV cache, GPU memory, system RAM, and context length should be considered together.
3. **Keep local services local by default.** Bind to localhost unless the user explicitly opts into LAN access.
4. **Prefer understandable, recoverable operations.** Provisioning should be repeatable, configuration writes should be safe, and server failures should produce actionable diagnostics.
5. **Keep the tool focused.** The panel coordinates inference engines rather than trying to reimplement them.

## Project status

The repository contains the application source, tests, architecture notes, deployment documentation, and GitHub Actions workflows. See the [README](README.md) for installation, supported backends, and development commands, and the [Roadmap](docs/ROADMAP.md) for project direction.
