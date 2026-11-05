# User Guide

This guide collects the detailed setup and workflow notes for Local LLM Panel. For prerequisites and the short first-run path, start with the [README](../README.md).

## First-run walkthrough

1. Open the app and go to **Dashboard**.
2. Choose **Provision WSL** to prepare vLLM in WSL2, or **Install llama.cpp** to set up the native Windows backend.
3. Open **Search** and find a model. Review the context, architecture, quantization variants, fit verdict, and estimated throughput.
4. Review the available variants and use **Pull** to download a model ahead of time. vLLM can also fetch Hugging Face models on first server start.
5. Go to **Servers → New server**, choose the backend, model, and task (`instruct` or `embed`), and configure context length, quantization, GPU utilization, port, and served model name.
6. Use **Context fit test** to check the planned context against current free GPU memory and available system RAM.
7. Start the server. Logs, health status, and metrics are available from the Dashboard and server panels.
8. Optional: enable the OpenAI-compatible gateway under **Settings → Advanced Mode → Network** and configure an API key.

## Understanding model-fit estimates

Fit scores are planning guidance, not a guarantee that a model will start or reach a particular speed. Results depend on the selected quantization, runtime overhead, KV-cache dtype, context length, other processes using the GPU, backend support, and actual hardware behavior.

The app estimates context and memory requirements from model configuration and architecture metadata where available. Unknown or incomplete model metadata can require conservative fallback assumptions. Measured throughput from actual runs should take precedence over estimated tok/s.

## vLLM in WSL2

The vLLM backend is provisioned inside an Ubuntu WSL2 distribution. Provisioning uses an app-managed virtual environment and is designed to be repeatable: running it again can repair or update the managed environment.

The panel checks that the NVIDIA driver and GPU are visible inside WSL. The WSL memory limit is controlled by the user's `.wslconfig`; the app displays relevant information but does not silently rewrite that file.

### FlashInfer sampler note

vLLM's FlashInfer top-k/top-p sampler may JIT-compile CUDA code on first use and can require `nvcc`, a C compiler, and `ninja` inside WSL.

The panel disables the FlashInfer sampler by default with `VLLM_USE_FLASHINFER_SAMPLER=0`, allowing vLLM to use its built-in PyTorch sampler. Users who specifically need the FlashInfer JIT path can use **Dashboard → FlashInfer JIT Ready → Check → Install CUDA Build Tools**. That optional toolkit installation downloads several GB.

When startup logs match known sampler or environment failures, the server card can display a targeted hint and restart option.

## Native llama.cpp

The native backend launches `llama-server.exe` on Windows and uses GGUF model files.

1. Open **Dashboard → Install llama.cpp**.
2. Select the desired fork/channel and install a Windows x64 CUDA build compatible with the NVIDIA driver.
3. Review install directories and the GGUF directory in **Settings**, or supply a custom `llama-server.exe` path.
4. Download GGUF files through **Search**, import a local GGUF through **Library**, or place files in the configured folder.
5. For split models, select the first shard. Add a compatible `mmproj` file when required for vision.
6. Create a llama.cpp server and review capability-gated options such as MoE CPU offload and `--fit` auto-fit.

Custom llama.cpp forks can be added under **Settings → llama.cpp Forks**. A fork definition identifies its GitHub repository and release asset naming convention; the app uses that information to discover supported Windows CUDA builds.

If GitHub release discovery is unavailable, use the fork's release page to download the Windows x64 CUDA archive and its matching CUDA runtime archive, extract them together, and select a custom executable path in Settings.

## Speculative decoding

Speculative decoding is optional and disabled by default. It runs a smaller draft model alongside the main instruct model to propose tokens that the target model verifies.

- **vLLM:** enter a Hugging Face model ID or local path for the draft model.
- **llama.cpp:** select a draft GGUF file.
- Choose the number of draft tokens to propose per verification step.
- The draft and target models must have compatible tokenizer/vocabulary and formatting.
- Draft weights and KV cache consume additional GPU/RAM capacity. Re-run the Context fit test before starting.

Backend support and command-line flags vary by engine version. The app checks installed binary capabilities and should report unsupported configurations rather than silently assuming a flag exists.

## OpenAI-compatible gateway

Enable the gateway under **Settings → Advanced Mode → Network** and configure an API key. The default endpoint is `http://127.0.0.1:11434/v1`.

All `/v1/*` routes require `Authorization: Bearer <key>`. The `/health` route remains open for liveness checks. The gateway supports:

- `GET /v1/models` and `GET /v1/models/{id}`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

Requests are routed by served model name or model ID to a compatible running backend. Requests that do not need compatibility normalization are forwarded without injecting defaults.

Example:

```bash
curl http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer $LOCAL_LLM_KEY"
```

### Client metadata and thinking controls

The model catalog includes context and capability metadata for client applications. Thinking/reasoning options are normalized across common request conventions, including `reasoning_effort`, `thinking`, Ollama's `think`, and `chat_template_kwargs`.

Some clients cache the model catalog when they connect. When server names, context limits, or capabilities change, refresh the client's model catalog or reconnect it.

## Configuration and data locations

| Data | Default location |
|---|---|
| Config, measured stats, benchmarks | `%APPDATA%\local-llm-panel\config.json` |
| Conversations | `%APPDATA%\local-llm-panel\conversations.json` |
| Native llama.cpp logs | `%APPDATA%\local-llm-panel\logs\<server-id>.log` |
| GGUF models | `%APPDATA%\local-llm-panel\gguf` |
| llama.cpp installs | `%APPDATA%\local-llm-panel\llama.cpp\<fork-id>` |
| vLLM virtual environment and logs | `~/llm-lp/.venv`, `~/llm-lp/logs` |
| Hugging Face cache | `~/.cache/huggingface` (or configured `HF_HOME`) |

Hugging Face and GitHub tokens are protected with Windows DPAPI. Tokens are redacted from configuration-export payloads.

## Troubleshooting notes

- **Build reports `lld-link.exe` missing:** use the npm build scripts. The build runner searches standard LLVM and Visual Studio locations and configures the linker.
- **Windows 11 24H2+ `0xc0000139` startup crash:** the Rust target configuration includes the `windows_legacy` getrandom backend workaround.
- **WSL cannot see the GPU:** verify that WSL2 is current and `nvidia-smi` works inside the Ubuntu distribution.
- **Model starts but has less context than expected:** compare the selected quantization, model KV-cache architecture, runtime overhead, and other GPU processes; use the Context fit test.
- **Gateway client shows stale models or token limits:** refresh the client's cached model catalog or reconnect.
- **llama.cpp flag rejected:** the installed fork/version may not implement that option. Reinstall a compatible build or disable the unsupported feature.

When reporting a problem, include the app version, Windows version, GPU and VRAM, backend, relevant server logs, and the steps to reproduce. Redact tokens and private data before sharing logs.
