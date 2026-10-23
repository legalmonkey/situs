"""Live end-to-end network test: Coordinator + 2 WebSocket Workers + OpenAI API streaming."""

import pytest
import asyncio
import httpx
import uvicorn
from contextlib import asynccontextmanager
from situs.config import WorkerConfig
from situs.coordinator.app import app, registry
from situs.worker.client import WorkerDaemon


@pytest.mark.asyncio
async def test_live_websocket_cluster_e2e():
    registry.clear()
    port = 8765
    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(server_config)

    # 1. Start Coordinator server task
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)

    worker_tasks = []
    try:
        # 2. Launch Worker A and Worker B daemons connecting over real WebSockets
        cfg_a = WorkerConfig(
            coordinator_url=f"ws://127.0.0.1:{port}/ws/worker",
            worker_id="worker-live-a",
            device="cpu",
            model_name="gemma3:4b",
            compression_dtype="int8",
        )
        cfg_b = WorkerConfig(
            coordinator_url=f"ws://127.0.0.1:{port}/ws/worker",
            worker_id="worker-live-b",
            device="cpu",
            model_name="gemma3:4b",
            compression_dtype="int8",
        )

        daemon_a = WorkerDaemon(cfg_a)
        daemon_b = WorkerDaemon(cfg_b)

        worker_tasks.append(asyncio.create_task(daemon_a.run()))
        worker_tasks.append(asyncio.create_task(daemon_b.run()))

        # Await WebSocket registrations
        for _ in range(30):
            if len(registry.get_alive_workers()) >= 2:
                break
            await asyncio.sleep(0.1)

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
            # 3. Check /status
            resp = await client.get("/status")
            assert resp.status_code == 200
            status_data = resp.json()
            assert status_data["active_workers_count"] >= 2
            assert "worker-live-a" in registry.workers
            assert "worker-live-b" in registry.workers

            # 4. Partition model across the two live workers
            part_resp = await client.post("/admin/partition?model_name=gemma3:4b&total_layers=8")
            assert part_resp.status_code == 200

            # Allow workers to initialize stage engines
            await asyncio.sleep(0.5)

            # 5. Send an OpenAI Chat Completion request
            chat_payload = {
                "model": "gemma3:4b",
                "messages": [{"role": "user", "content": "Hi!"}],
                "max_tokens": 4,
                "stream": True,
            }

            received_chunks = []
            async with client.stream("POST", "/v1/chat/completions", json=chat_payload) as stream_resp:
                assert stream_resp.status_code == 200
                async for line in stream_resp.aiter_lines():
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        received_chunks.append(line)

            # Assert that streamed SSE chunks arrived across the two-worker pipeline
            assert len(received_chunks) > 0

    finally:
        # Cleanup
        for t in worker_tasks:
            t.cancel()
        server.should_exit = True
        await server_task
