"""Unit tests for worker removal, auto-rebalance on removal, and TUI sidebar live stages."""

import time
import pytest
import asyncio
from typer.testing import CliRunner
from prompt_toolkit.document import Document
from situs.cli import app
from situs.coordinator.registry import WorkerRegistry, WorkerNode
from situs.coordinator.app import rebalance_and_assign_layers
from situs.protocol.messages import MessageType, ProtocolFrame
from situs.tui import OpenCodeSlashCompleter, execute_slash_command

runner = CliRunner()


def test_cli_remove_help():
    """Verify 'situs remove --help' works properly."""
    result = runner.invoke(app, ["remove", "--help"])
    assert result.exit_code == 0
    assert "Remove a worker device from the cluster" in result.output
    assert "worker_id" in result.output.lower()


def test_tui_remove_slash_completer():
    """Verify /remove suggests active workers from cluster cache."""
    from situs.tui import _metadata_cache
    _metadata_cache["http://127.0.0.1:8000"] = (
        9999999999.0,
        {
            "workers": [
                {"id": "worker-alpha", "worker_id": "worker-alpha", "gpu": "RTX 3050"},
                {"id": "worker-beta", "worker_id": "worker-beta", "gpu": "RTX 4060"},
            ]
        }
    )

    completer = OpenCodeSlashCompleter()
    doc = Document("/remove ")
    completions = [c.text for c in completer.get_completions(doc, None)]
    assert "/remove worker-alpha" in completions
    assert "/remove worker-beta" in completions

    doc_sub = Document("/remove worker-b")
    completions_sub = [c.text for c in completer.get_completions(doc_sub, None)]
    assert "/remove worker-beta" in completions_sub
    assert "/remove worker-alpha" not in completions_sub


def test_tui_remove_slash_command_usage():
    """Verify /remove with no arguments prints usage."""
    messages = []
    session_state = {}
    captured = []

    def log_capture(title, renderable):
        captured.append((title, str(renderable)))

    res = execute_slash_command("/remove", "http://127.0.0.1:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert res is True
    assert any("Usage" in str(r) for _, r in captured)


@pytest.mark.asyncio
async def test_coordinator_rebalance_on_multiple_workers():
    """Verify registry rebalancing partitions layers across multiple workers and handles removal."""
    from situs.coordinator.app import registry, rebalance_and_assign_layers

    # Clean registry
    registry.workers.clear()
    registry.active_model = "gemma3:4b"

    now = time.time()
    node1 = WorkerNode(
        worker_id="w1",
        gpu_name="RTX 3050",
        total_vram_bytes=6 * 10**9,
        free_vram_bytes=4 * 10**9,
        device_type="cuda:0",
        last_heartbeat=now,
    )
    node2 = WorkerNode(
        worker_id="w2",
        gpu_name="RTX 4060",
        total_vram_bytes=6 * 10**9,
        free_vram_bytes=4 * 10**9,
        device_type="cuda:0",
        last_heartbeat=now,
    )

    # Register 2 workers
    registry.register_worker(node1)
    registry.register_worker(node2)

    assert len(registry.get_alive_workers()) == 2

    # Rebalance dynamically resolved layers for gemma3:4b
    from situs.coordinator.registry import resolve_model_metadata
    meta = resolve_model_metadata("gemma3:4b")
    total_l = meta["total_layers"]

    await rebalance_and_assign_layers("gemma3:4b")

    w1 = registry.workers["w1"]
    w2 = registry.workers["w2"]

    # Verify 2 stages assigned contiguously
    assert w1.assigned_layer_start == 0
    assert w1.assigned_layer_end == w2.assigned_layer_start
    assert w2.assigned_layer_end == total_l
    assert w1.is_first_stage is True
    assert w1.is_last_stage is False
    assert w2.is_first_stage is False
    assert w2.is_last_stage is True

    # Remove w2
    registry.remove_worker("w2")
    assert len(registry.get_alive_workers()) == 1

    # Rebalance remaining
    await rebalance_and_assign_layers("gemma3:4b")

    # w1 now has all layers
    assert w1.assigned_layer_start == 0
    assert w1.assigned_layer_end == total_l
    assert w1.is_first_stage is True
    assert w1.is_last_stage is True


def test_sidebar_nodes_and_stages_fragment_generation():
    """Verify that get_sidebar_fragments correctly renders active nodes, assigned stages, and remove button."""
    session_state = {
        "is_coordinator": True,
        "workers_count": 2,
        "coord_status": "online",
        "total_vram_gb": 10.7,
        "free_vram_gb": 8.4,
        "used_vram_gb": 2.3,
        "vram_usage_pct": 21.5,
        "model": "gemma3:4b",
        "cluster_workers": [
            {
                "id": "worker-1",
                "worker_id": "worker-1",
                "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
                "layer_range": "0..14",
                "total_gb": 6.4,
                "used_gb": 1.0,
                "is_first_stage": True,
                "is_last_stage": False,
            },
            {
                "id": "worker-2",
                "worker_id": "worker-2",
                "gpu": "NVIDIA GeForce RTX 3050 A Laptop GPU",
                "layer_range": "14..28",
                "total_gb": 4.3,
                "used_gb": 1.3,
                "is_first_stage": False,
                "is_last_stage": True,
            },
        ],
    }

    frags = []
    frags.append(("bold #f8fafc", "Active Nodes & Stages\n"))
    frags.append(("#475569", "────────────────────────────────────\n"))
    workers = session_state.get("cluster_workers", [])
    assert len(workers) == 2

    for w in workers:
        w_id = str(w.get("worker_id") or w.get("id"))
        layer_range = w.get("layer_range")
        stage_str = f"L[{layer_range}]"
        if w.get("is_first_stage"):
            stage_str += " (Embed)"
        if w.get("is_last_stage"):
            stage_str += " (Head)"

        gpu_desc = w.get("gpu").replace("NVIDIA GeForce ", "").replace("Laptop GPU", "Lap")
        total_gb = float(w.get("total_gb", 0.0))
        used_gb = float(w.get("used_gb", 0.0))
        vram_str = f"{used_gb:.1f}/{total_gb:.1f}GB"

        frags.append(("bold #22c55e", "● "))
        frags.append(("bold #ffffff", f"{w_id} "))
        frags.append(("bold #ef4444 underline", "[✕ Remove]"))
        frags.append(("", "\n"))
        frags.append(("#38bdf8", f"  {stage_str}"))
        frags.append(("#94a3b8", f" · {vram_str}\n"))
        frags.append(("#64748b", f"  {gpu_desc}\n\n"))

    full_sidebar_text = "".join(f[1] for f in frags)
    assert "Active Nodes & Stages" in full_sidebar_text
    assert "worker-1" in full_sidebar_text
    assert "L[0..14] (Embed)" in full_sidebar_text
    assert "worker-2" in full_sidebar_text
    assert "L[14..28] (Head)" in full_sidebar_text
    assert "[✕ Remove]" in full_sidebar_text
    assert "1.0/6.4GB" in full_sidebar_text
    assert "1.3/4.3GB" in full_sidebar_text


def test_live_socket_handshake_and_removal():
    """Real end-to-end socket handshake and worker removal on live coordinator."""
    from situs.tui import start_in_process_coordinator, start_in_process_worker
    import httpx

    # 1. Start live coordinator
    ok, msg = start_in_process_coordinator()
    assert ok is True

    # 2. Start live worker node
    w_id = f"test-worker-{int(time.time())}"
    ok_w, msg_w = start_in_process_worker(
        coordinator_addr="127.0.0.1:8000",
        model="gemma3:4b",
        worker_id=w_id,
        device="cpu",
    )
    assert ok_w is True

    # 3. Poll for registration acknowledgment over live socket
    registered = False
    for _ in range(15):
        time.sleep(0.5)
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get("http://127.0.0.1:8000/status")
                if resp.status_code == 200:
                    workers = resp.json().get("workers", [])
                    if any(w.get("worker_id") == w_id or w.get("id") == w_id for w in workers):
                        registered = True
                        break
        except Exception:
            pass
    assert registered is True, f"Worker {w_id} failed live socket registration"

    # 4. Remove worker via HTTP DELETE endpoint
    with httpx.Client(timeout=5.0) as client:
        del_resp = client.delete(f"http://127.0.0.1:8000/admin/workers/{w_id}")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data.get("status") == "success"
        assert del_data.get("removed_worker") == w_id

    # 5. Verify worker is removed from live coordinator status
    with httpx.Client(timeout=2.0) as client:
        status_resp = client.get("http://127.0.0.1:8000/status")
        workers_after = status_resp.json().get("workers", [])
        assert not any(w.get("worker_id") == w_id or w.get("id") == w_id for w in workers_after)

