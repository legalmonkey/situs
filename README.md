# Situs: Distributed GPU Pooling Across the Internet

**Situs** is a distributed LLM inference runtime that partitions causal language models across independent consumer GPUs in different homes over the public internet, exposing them through a single unified OpenAI-compatible API.

---

## Key Highlights

- **Pipeline Parallelism over WAN**: Minimizes network traffic by partitioning contiguous transformer layers across machines. Avoids tensor parallelism's frequent all-reduce synchronization barriers.
- **Progressive JIT Layer Streaming & Dynamic Hot-Swapping (Strategy 3)**: Enables progressive layer assembly on low-bandwidth networks (e.g., 2 MB/s). Initializes genuine transformer layers dynamically in-place as verified weights stream in the background without dropping active sessions.
- **Wire Quantization Decoupling & Zstandard Binary Framing (Strategy 4)**: Compresses network layer distribution with streaming Zstandard compression and BitNet 1.6 bpw packing, framed over raw 22-byte binary packets (`b"SITC"` header), eliminating Base64 wire overhead and shrinking weight transfer sizes by over 75%.
- **Zero KV-Cache Network Transfer**: Each worker stores KV caches strictly for its own assigned layers. KV caches *never* cross the network.
- **Asymmetric Ring Token Loop**:
  - Worker A $\to$ Worker B: Compressed intermediate hidden state (~4 KB in INT8).
  - Worker B $\to$ Worker A: Sampled `token_id` (4 bytes).
- **Zero Port Forwarding**: Workers initiate outbound reverse connections over TLS/WebSockets to the Coordinator Relay. Works through residential NATs, CGNAT, and firewalls with no router configuration.
- **Dynamic Activation Quantization & Temporal Deltas**: Quantizes boundary activations to INT8 dynamically and applies DisTrO-style temporal delta sparsification, reducing token transfers from 4.1 KB down to < 500 bytes.
- **Drafterless Speculative Decoding**: Uses prompt and history n-gram lookup on Worker A to draft multi-token candidate bursts, verified in a single forward pass on Worker B ($2.0\times - 3.2\times$ effective speedup).
- **Direct P2P WebRTC DataChannels**: Performs STUN UDP hole-punching via Google STUN servers to bypass the Coordinator relay, cutting latency by $50\%$ ($1 \times \text{RTT}$ vs $2 \times \text{RTT}$) with automatic WebSocket fallback.
- **OpenAI Standard Gateway**: Drop-in compatible with `openai` SDK, Cursor, Continue.dev, LangChain, LiteLLM, Open WebUI, and standard OpenAI clients.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                        CLIENT / OPENAI API CALLER                                 |
|                     POST /v1/chat/completions (streaming)                         |
+-----------------------------------------------------------------------------------+
                                        | (HTTP / SSE Stream)
                                        v
+-----------------------------------------------------------------------------------+
|                           COORDINATOR & RELAY                                     |
|  - Worker Registry & Health Heartbeats                                            |
|  - Request Orchestration & Token Streaming                                        |
|  - Duplex Outbound Tunnel Gateway (zero router port forwarding required)          |
+-----------------------------------------------------------------------------------+
           |                                                        ^
           | 1. Prompt / Token IDs                                  | 4. Next Token IDs
           v                                                        |    (Streamed to user)
+------------------------------------+    Activation Transfer       +------------------------------------+
|       WORKER A (Home GPU 1)        | ---------------------------> |       WORKER B (Home GPU 2)        |
| - Tokenizer & embed_tokens         |   Shape: [1, 1, hidden_dim]  | - transformer_layers [16..31]      |
| - transformer_layers [0..15]       |   Dynamic INT8 Quantization  | - LayerNorm & lm_head              |
| - Local KV-Cache (Layers 0..15)    |   Outbound WebSocket Stream  | - Local KV-Cache (Layers 16..31)   |
| - Partial Weight Loader            |                              | - Token Sampler (top-p / temp)     |
+------------------------------------+                              +------------------------------------+
           ^                                                                     |
           +------------------------- 3. Next Token ID --------------------------+
                              (Only 4 bytes: int32 token_id!)
```

---

## CLI Quickstart (PowerShell / macOS / Linux)

Situs comes with a streamlined terminal interface like OpenCode and Claude Code.

### 1. Installation & Environment Check

```powershell
git clone https://github.com/legalmonkey/situs.git
cd situs
pip install -e .

# Run automatic system & GPU hardware diagnostics:
situs setup
# (or in PowerShell without pip PATH: .\situs.ps1 setup)
```

---

### 2. Launch the Coordinator (Laptop 1)

```powershell
situs coordinator --port 8000
```
*Prints your local IP (`172.20.164.173`), worker connection URL, and OpenAI endpoints.*

---

### 3. Join the Cluster (Laptop 2 & Laptop 1)

On each laptop, join the cluster with a single command (auto-detects GPU hardware, VRAM, and enables P2P & speculative acceleration):

```powershell
# On Laptop 1:
situs join 127.0.0.1:8000 --id worker-a

# On Laptop 2 (replace with Laptop 1's Wi-Fi IP):
situs join 172.20.164.173:8000 --id worker-b
```

---

### 4. Check Cluster Status & Partition

```powershell
# View registered GPUs, VRAM, and pipeline stages in a live table:
situs status

# Auto-partition model across active workers:
situs partition --model llama-3.1:8b-instruct-q8_0 --layers 32
```

---

### 5. OpenCode-Style Interactive TUI & Slash Commands

Run `situs` with no arguments (or `situs chat`) in PowerShell:

```powershell
situs
```

This launches the full OpenCode interface:
- **Centered pixel logo** and floating input card with model & GPU cluster badges.
- **Real-time `/` Slash Command Palette**: Typing `/` immediately displays an interactive completion dropdown:
  - `/status`: Inspect connected GPUs, VRAM allocations, and pipeline stages.
  - `/partition`: Rebalance and partition layers across connected GPUs.
  - `/nodes`: Inspect worker node health, IP addresses, and WebRTC P2P latency.
  - `/doctor`: Run system & GPU hardware diagnostics.
  - `/model`: Inspect model architecture and layer depth.
  - `/p2p`: Toggle WebRTC direct P2P vs WebSocket relay.
  - `/speculative`: Toggle prompt-lookup speculative decoding.
  - `/clear`: Clear conversation memory.
  - `/system`: Update system instructions.
  - `/help`: Display slash command cheat sheet.
  - `/exit`: Exit the session.
- **Streaming Markdown Tokens**: Real-time token generation with live tokens/sec and latency stats.
- **Bottom Status Bar**: Shows `~:main` (git branch), shortcuts (`tab nodes  / commands  ctrl+c exit`), and version.

For one-shot scripting in PowerShell:
```powershell
situs run "Explain distributed pipeline parallelism over home Wi-Fi"
```

---

## 5. Send Inference Requests

### Option A: Standard OpenAI Python SDK
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("SITUS_API_KEY", "your-api-key"),
)

response = client.chat.completions.create(
    model="llama-3.1:8b-instruct-q8_0",
    messages=[{"role": "user", "content": "Explain pipeline parallelism in two sentences."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Option B: cURL Streaming (Server-Sent Events)
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1:8b-instruct-q8_0",
    "messages": [{"role": "user", "content": "What is distributed inference?"}],
    "stream": true
  }'
```

---

## Cluster Security & Authentication

Situs enforces security and authentication in both local and production environments:

### Environment Configuration

| Variable | Description | Default | Production Requirement |
| :--- | :--- | :--- | :--- |
| `SITUS_ENV` | Environment mode (`development` / `production`) | `development` | Optional (set to `production` for strict enforcement) |
| `SITUS_WORKER_AUTH_TOKEN` | Secret token required for workers to join the cluster | `None` (open dev mode) | **Required** in production |
| `SITUS_API_KEY` | Bearer token required for OpenAI API endpoints (`/v1/*`) | `None` (open dev mode) | **Required** in production |
| `SITUS_ADMIN_KEY` | Secret key required for admin control (`/admin/*`) | `None` (open dev mode) | **Required** in production |
| `SITUS_ALLOWED_ORIGINS` | Comma-separated explicit CORS allowed origins | `localhost, 127.0.0.1` | Explicit origins required; wildcard `*` is prohibited |

### Running with Security Enabled

1. **Launch Coordinator with Authentication**:
   ```powershell
   situs coordinator --port 8000 --token "your-cluster-secret" --api-key "client-api-key" --admin-key "admin-secret-key"
   # Or configure via environment variables:
   $env:SITUS_WORKER_AUTH_TOKEN="your-cluster-secret"
   $env:SITUS_API_KEY="client-api-key"
   $env:SITUS_ADMIN_KEY="admin-secret-key"
   situs coordinator --port 8000
   ```

2. **Join Worker with Authentication**:
   ```powershell
   situs join 172.20.164.173:8000 --id worker-b --token "your-cluster-secret"
   ```

3. **CORS Security**:
   Wildcard origins (`*`) are strictly prohibited when credentials are enabled to prevent cross-origin credential theft. Set explicit origins in `SITUS_ALLOWED_ORIGINS`.

---

## Running the Automated Test Suite

```bash
pytest situs/tests -v
```
