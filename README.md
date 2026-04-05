# self-evolving-agent

## 中文说明

### 拓扑

- 本地机器运行控制器逻辑，核心代码在 `client/controller.py`。
- `server1` 是远端大模型大脑，对外暴露 OpenAI 兼容的 `POST /v1/chat/completions` 接口。
- `server2` 是远端评测服务，对外暴露普通 HTTP 接口：
  - `POST /eval/jobs`
  - `GET /eval/jobs/{job_id}`
- `bridge/client_ws_proxy.py` 是你本地到远端 `server1` 的桥接脚本，适合处理大模型长回复和超时问题。

### 标签映射

- `A` 表示正常
- `B-J` 表示攻击
- 输出如果不在 `A-J` 内，也按正常处理

### 启动顺序

1. 在 `server1` 所在侧启动桥接后的模型代理：

   ```bash
   export VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
   uvicorn server1.server_ws_proxy:app --host 0.0.0.0 --port 8000
   ```

2. 在 `server2` 所在侧启动评测服务：

   ```bash
   uvicorn server2.eval_service:app --host 0.0.0.0 --port 19000
   ```

3. 在你本地运行控制器或你自己的驱动脚本，并把它指向两个远端基础地址：

   ```bash
   export SERVER1_BASE_URL="http://<server1-host>:8000"
   export SERVER2_BASE_URL="http://<server2-host>:19000"
   ```

目前控制器主流程还是库代码，最小闭环入口是 `client/controller.py` 里的 `run_iteration(...)`。

### bridge 配置

`bridge/client_ws_proxy.py` 通过环境变量读取桥接配置：

- `REMOTE_WSS_URL`
- `REMOTE_ORIGIN`
- `REMOTE_REFERER`
- `LOCAL_HOST`
- `LOCAL_PORT`
- `MODEL_NAME`

示例：

```bash
export REMOTE_WSS_URL="wss://your-server1-proxy/ws"
export REMOTE_ORIGIN="https://your-server1-origin"
export REMOTE_REFERER="https://your-server1-referer"
export MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
python bridge/client_ws_proxy.py
```

## Topology

- The local machine runs the controller logic from `client/controller.py`.
- `server1` is the remote LLM brain exposed as an OpenAI-compatible `POST /v1/chat/completions` endpoint.
- `server2` is the remote evaluator exposed over ordinary HTTP with `POST /eval/jobs` and `GET /eval/jobs/{job_id}`.

## Label Mapping

- `A` means normal.
- `B` through `J` mean attack.
- Any output outside `A-J` counts as normal.

## Startup Order

1. Start the `server1` bridge that fronts your remote model endpoint. In this repo that bridge is `server1/server_ws_proxy.py`.
   ```bash
   export VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
   uvicorn server1.server_ws_proxy:app --host 0.0.0.0 --port 8000
   ```
2. Start the evaluator service on the remote side.
   ```bash
   uvicorn server2.eval_service:app --host 0.0.0.0 --port 19000
   ```
3. Run the local controller process or driver on your machine, pointing it at the two base URLs with `SERVER1_BASE_URL` and `SERVER2_BASE_URL`.

The controller itself is library code today, so the entrypoint is whatever local runner imports `run_iteration` from `client/controller.py`.

## Bridge Configuration

`bridge/client_ws_proxy.py` now reads its remote bridge settings from environment variables:

- `REMOTE_WSS_URL`
- `REMOTE_ORIGIN`
- `REMOTE_REFERER`
- `LOCAL_HOST`
- `LOCAL_PORT`
- `MODEL_NAME`

Example:

```bash
export REMOTE_WSS_URL="wss://your-server1-proxy/ws"
export REMOTE_ORIGIN="https://your-server1-origin"
export REMOTE_REFERER="https://your-server1-referer"
export MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
python bridge/client_ws_proxy.py
```
