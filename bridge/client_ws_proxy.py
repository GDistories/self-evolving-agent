import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import uvicorn


# =========================
# 配置
# =========================
COOKIE_TXT_PATH = Path("cookie.txt")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(Path(__file__).with_name(".env"))

REMOTE_WSS_URL = os.getenv(
    "REMOTE_WSS_URL",
    "wss://XXX.com/5eb45d6a-11e4-4dbe-a98c-6ef08c533902/proxy/9000/ws",
)

ORIGIN = os.getenv("REMOTE_ORIGIN", "XXX.com")
REFERER = os.getenv(
    "REMOTE_REFERER",
    "XXX.com/5eb45d6a-11e4-4dbe-a98c-6ef08c533902/lab",
)

LOCAL_HOST = os.getenv("LOCAL_HOST", "127.0.0.1")
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "18000"))

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")

# 控制台输出控制
PRINT_STREAM_TO_CONSOLE = True
PRINT_CONTENT_TO_CONSOLE = True
PRINT_REASONING_TO_CONSOLE = True

# 区分正文和 reasoning 的前缀
CONTENT_PREFIX = "[content] "
REASONING_PREFIX = "[reasoning] "

# 是否在 content/reasoning 类型切换时自动换行，避免粘在一起难看
AUTO_NEWLINE_WHEN_STREAM_TYPE_SWITCH = True

WS_PING_INTERVAL = 15
WS_OPEN_TIMEOUT = 30
WS_REPLY_TIMEOUT: Optional[float] = None


app = FastAPI(title="Local OpenAI WS Bridge")


def now_ts() -> int:
    return int(time.time())


def make_chat_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:18]


def read_cookie_value(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"cookie txt 不存在: {path}")
    cookie = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not cookie:
        raise ValueError("cookie txt 为空")
    return cookie


def build_ws_headers() -> List[Tuple[str, str]]:
    cookie_value = read_cookie_value(COOKIE_TXT_PATH)
    headers = {
        "Cookie": cookie_value,
        "Origin": ORIGIN,
        "Referer": REFERER,
        "User-Agent": "Mozilla/5.0",
    }
    return list(headers.items())


def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_content_and_reasoning_from_openai_obj(obj: dict) -> tuple[str, str]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    choices = obj.get("choices") or []
    for choice in choices:
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}

        # content
        if isinstance(delta.get("content"), str) and delta["content"]:
            content_parts.append(delta["content"])
        elif isinstance(message.get("content"), str) and message["content"]:
            content_parts.append(message["content"])

        # reasoning
        if isinstance(delta.get("reasoning"), str) and delta["reasoning"]:
            reasoning_parts.append(delta["reasoning"])
        elif isinstance(message.get("reasoning"), str) and message["reasoning"]:
            reasoning_parts.append(message["reasoning"])

    return "".join(content_parts), "".join(reasoning_parts)


def build_final_chat_completion(
    model_name_seen: str,
    collected_content: list[str],
    collected_reasoning: list[str],
    usage_obj: Optional[dict],
) -> dict:
    final_message = {
        "role": "assistant",
        "content": "".join(collected_content),
    }

    if collected_reasoning:
        final_message["reasoning"] = "".join(collected_reasoning)

    return {
        "id": make_chat_id(),
        "object": "chat.completion",
        "created": now_ts(),
        "model": model_name_seen,
        "choices": [
            {
                "index": 0,
                "message": final_message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage_obj,
    }


async def ws_ping_sender(ws):
    try:
        while True:
            await asyncio.sleep(WS_PING_INTERVAL)
            await ws.ping()
    except Exception:
        return


def console_print_piece(
    piece_type: str,
    text: str,
    last_print_type: Optional[str],
) -> str:
    """
    piece_type: "reasoning" 或 "content"
    返回新的 last_print_type
    """
    if not PRINT_STREAM_TO_CONSOLE or not text:
        return last_print_type or ""

    if piece_type == "reasoning" and not PRINT_REASONING_TO_CONSOLE:
        return last_print_type or ""
    if piece_type == "content" and not PRINT_CONTENT_TO_CONSOLE:
        return last_print_type or ""

    prefix = REASONING_PREFIX if piece_type == "reasoning" else CONTENT_PREFIX

    need_prefix = last_print_type != piece_type
    need_newline = (
        AUTO_NEWLINE_WHEN_STREAM_TYPE_SWITCH
        and last_print_type is not None
        and last_print_type != piece_type
    )

    out = ""
    if need_newline:
        out += "\n"
    if need_prefix:
        out += prefix
    out += text

    print(out, end="", flush=True)
    return piece_type


async def call_remote_ws(payload: dict, stream: bool) -> AsyncGenerator[str, None]:
    headers = build_ws_headers()

    async with websockets.connect(
        REMOTE_WSS_URL,
        additional_headers=headers,
        open_timeout=WS_OPEN_TIMEOUT,
        ping_interval=None,
        max_size=None,
    ) as ws:
        ping_task = asyncio.create_task(ws_ping_sender(ws))
        collected_content: List[str] = []
        collected_reasoning: List[str] = []
        usage_obj: Optional[dict] = None
        model_name_seen = payload.get("model", MODEL_NAME)

        # 记录控制台上一次打印的流类型，用来决定是否加前缀/换行
        last_print_type: Optional[str] = None

        try:
            await ws.send(json.dumps(payload, ensure_ascii=False))

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=WS_REPLY_TIMEOUT)
                except websockets.exceptions.ConnectionClosed:
                    if not stream and (collected_content or collected_reasoning):
                        final_obj = build_final_chat_completion(
                            model_name_seen=model_name_seen,
                            collected_content=collected_content,
                            collected_reasoning=collected_reasoning,
                            usage_obj=usage_obj,
                        )
                        yield json.dumps(final_obj, ensure_ascii=False)
                        return
                    raise

                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", errors="replace")

                obj = safe_json_loads(msg)
                if not obj:
                    continue

                msg_type = obj.get("type")

                if msg_type == "ping":
                    continue

                if msg_type == "upstream_status":
                    status = obj.get("status_code")
                    if status != 200:
                        if PRINT_STREAM_TO_CONSOLE:
                            print(f"\n[警告] 收到上游异常状态码: {status}，正在读取详细原因...")
                    continue

                if msg_type == "error":
                    body = obj.get("body") or obj.get("message") or str(obj)
                    if not stream and (collected_content or collected_reasoning):
                        final_obj = build_final_chat_completion(
                            model_name_seen=model_name_seen,
                            collected_content=collected_content,
                            collected_reasoning=collected_reasoning,
                            usage_obj=usage_obj,
                        )
                        yield json.dumps(final_obj, ensure_ascii=False)
                        return
                    raise RuntimeError(f"远端 ws_proxy 返回错误: {body}")

                if msg_type == "chunk":
                    data_line = obj.get("data", "")
                    if not isinstance(data_line, str):
                        continue
                    if not data_line.startswith("data:"):
                        continue

                    sse_payload = data_line[len("data:"):].strip()
                    if sse_payload == "[DONE]":
                        if stream:
                            yield "data: [DONE]\n\n"
                        else:
                            final_obj = build_final_chat_completion(
                                model_name_seen=model_name_seen,
                                collected_content=collected_content,
                                collected_reasoning=collected_reasoning,
                                usage_obj=usage_obj,
                            )
                            yield json.dumps(final_obj, ensure_ascii=False)
                        return

                    inner = safe_json_loads(sse_payload)
                    if inner is None:
                        continue

                    model_name_seen = inner.get("model", model_name_seen)
                    usage_obj = inner.get("usage") or usage_obj

                    content_piece, reasoning_piece = extract_content_and_reasoning_from_openai_obj(inner)

                    # 保持上游原始到达顺序：
                    # 如果这个 chunk 同时带 reasoning 和 content，就先打印 reasoning 再打印 content
                    # 因为通常模型是“思考 -> 输出正文”的语义。
                    if reasoning_piece:
                        collected_reasoning.append(reasoning_piece)
                        last_print_type = console_print_piece(
                            piece_type="reasoning",
                            text=reasoning_piece,
                            last_print_type=last_print_type,
                        )

                    if content_piece:
                        collected_content.append(content_piece)
                        last_print_type = console_print_piece(
                            piece_type="content",
                            text=content_piece,
                            last_print_type=last_print_type,
                        )

                    if stream:
                        yield f"data: {json.dumps(inner, ensure_ascii=False)}\n\n"

                if msg_type == "done":
                    if PRINT_STREAM_TO_CONSOLE and (PRINT_CONTENT_TO_CONSOLE or PRINT_REASONING_TO_CONSOLE):
                        print("", flush=True)

                    if not stream:
                        final_obj = build_final_chat_completion(
                            model_name_seen=model_name_seen,
                            collected_content=collected_content,
                            collected_reasoning=collected_reasoning,
                            usage_obj=usage_obj,
                        )
                        yield json.dumps(final_obj, ensure_ascii=False)
                    else:
                        yield "data: [DONE]\n\n"
                    return

        finally:
            ping_task.cancel()


@app.get("/health")
async def health():
    return {"ok": True, "remote_wss": REMOTE_WSS_URL, "model": MODEL_NAME}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": now_ts(),
                "owned_by": "local-ws-bridge",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON object")

    if not body.get("model"):
        body["model"] = MODEL_NAME

    stream = bool(body.get("stream", False))

    if stream:
        async def event_stream():
            try:
                async for chunk in call_remote_ws(body, stream=True):
                    yield chunk
            except Exception as e:
                err_obj = {
                    "error": {
                        "message": str(e),
                        "type": "bridge_error",
                        "code": 500,
                    }
                }
                yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        final_json = None
        async for piece in call_remote_ws(body, stream=False):
            final_json = piece

        if not final_json:
            raise RuntimeError("未收到任何返回")

        return JSONResponse(content=json.loads(final_json))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return PlainTextResponse(
        "Local OpenAI WS Bridge is running.\n"
        f"POST http://{LOCAL_HOST}:{LOCAL_PORT}/v1/chat/completions\n"
        f"GET  http://{LOCAL_HOST}:{LOCAL_PORT}/v1/models\n"
    )


if __name__ == "__main__":
    print("启动本地 OpenAI 兼容桥接服务")
    print(f"本地地址: http://{LOCAL_HOST}:{LOCAL_PORT}")
    print(f"远端 WSS: {REMOTE_WSS_URL}")
    print(f"Cookie txt: {COOKIE_TXT_PATH}")
    uvicorn.run(app, host=LOCAL_HOST, port=LOCAL_PORT)
