# Situs: Distributed GPU Pooling Across the Internet

**Situs** is a distributed LLM inference runtime that partitions causal language models across independent consumer GPUs in different homes over the public internet, exposing them through a single unified OpenAI-compatible API.

---

## Key Highlights

- **Pipeline Parallelism over WAN**: Minimizes network traffic by partitioning contiguous transformer layers across machines. Avoids tensor parallelism's frequent all-reduce synchronization barriers.
- **Fast-Boot Skeletons & Dynamic Hot-Swapping (Strategy 3)**: Enables instant cold start in < 1 second on low-bandwidth networks (e.g., 2 MB/s). Initializes shallow draft blocks with identity residual projections while high-fidelity full-precision layers stream in the background and hot-swap dynamically in-place without dropping active sessions.
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
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="situs")

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

## Running the Automated Test Suite

```bash
pytest situs/tests -v
```
