import asyncio
import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

VLLM_URL = os.getenv(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)


async def ping_loop(ws: WebSocket, interval: float = 10.0) -> None:
    """Periodically send heartbeat messages so upstream proxies don't think the connection is idle."""
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json({"type": "ping"})
    except Exception:
        # Client disconnected or socket closed
        return


@app.websocket("/ws")
async def websocket_proxy(ws: WebSocket) -> None:
    await ws.accept()
    ping_task = asyncio.create_task(ping_loop(ws, interval=10.0))

    try:
        # 先收客户端发来的请求 JSON
        req_text = await ws.receive_text()
        payload: dict[str, Any] = json.loads(req_text)

        # print("payload from client =", json.dumps(payload, ensure_ascii=False, indent=2))

        # 强制流式，便于边读边转发
        payload["stream"] = True

        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                VLLM_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
            ) as resp:
                # 先把状态发给前端
                await ws.send_json(
                    {
                        "type": "upstream_status",
                        "status_code": resp.status_code,
                    }
                )

                if resp.status_code != 200:
                    body = await resp.aread()
                    await ws.send_json(
                        {
                            "type": "error",
                            "status_code": resp.status_code,
                            "body": body.decode("utf-8", errors="replace"),
                        }
                    )
                    return

                # vLLM 流式接口会返回 SSE 风格的 data: ...
                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # 透传原始行，前端自己解析
                    await ws.send_json({"type": "chunk", "data": line})

                await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        ping_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
