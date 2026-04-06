# self-evolving-agent

## 中文说明

### 拓扑

- 本地机器运行控制器逻辑，核心代码在 `client/controller.py`。
- `server1` 是远端大模型大脑，对外暴露 OpenAI 兼容的 `POST /v1/chat/completions` 接口。
- `server2` 是远端评测服务，对外暴露普通 HTTP 接口：
  - `POST /eval/jobs`
  - `GET /eval/jobs/{job_id}`
- 本地桥接层分成两个独立进程：
  - `bridge/server1/bridge.py`
  - `bridge/server2/bridge.py`
- `bridge/server1/` 和 `bridge/server2/` 各自使用单独的 `cookie.txt`。
- 本地客户端只访问本地桥接 URL，不直接连远端服务。

### 标签映射

- `A` 表示正常
- `B-J` 表示攻击
- 输出如果不在 `A-J` 内，也按正常处理

### 启动顺序

1. 先启动 `server1`。

   ```bash
   cat > server1/.env <<'EOF'
   VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
   EOF
   uvicorn server1.server_ws_proxy:app --host 0.0.0.0 --port 8000
   ```

2. 再启动 `server2`。

   ```bash
   uvicorn server2.eval_service:app --host 0.0.0.0 --port 19000
   ```

3. 在本地启动 `bridge/server1/bridge.py`。

   ```bash
   cat > bridge/server1/.env <<'EOF'
   REMOTE_WSS_URL=wss://<server1-host>/v1/chat/completions
   REMOTE_ORIGIN=https://<server1-origin>
   REMOTE_REFERER=https://<server1-referer>
   LOCAL_HOST=127.0.0.1
   LOCAL_PORT=18000
   MODEL_NAME=Qwen/Qwen3.5-35B-A3B
   COOKIE_FILE=cookie.txt
   EOF
   python bridge/server1/bridge.py
   ```

4. 在本地启动 `bridge/server2/bridge.py`。

   ```bash
   cat > bridge/server2/.env <<'EOF'
   REMOTE_BASE_URL=https://<server2-host>/proxy/19000
   REMOTE_ORIGIN=https://<server2-origin>
   REMOTE_REFERER=https://<server2-referer>
   LOCAL_HOST=127.0.0.1
   LOCAL_PORT=19000
   COOKIE_FILE=cookie.txt
   REQUEST_TIMEOUT_SECONDS=30
   EOF
   python bridge/server2/bridge.py
   ```

5. 运行本地客户端 runner。

   ```bash
   cat > client/.env <<'EOF'
   SERVER1_BASE_URL=http://127.0.0.1:18000
   SERVER2_BASE_URL=http://127.0.0.1:19000
   POLL_INTERVAL_SECONDS=2.0
   EOF
   ```

`client/.env` 只控制运行时 URL 和轮询间隔。实验本身放在 `client/experiment.json`：

- `baseline_candidate` 负责 baseline candidate、prompt text、`candidate_id`
- `best_metrics` 负责初始最优指标
- `metric_config` 负责 metric policy 和数据集路径，包括 `primary_metric`、`min_value`、`tp_path`、`tn_path`
- `runner` 负责 `brain_model`、`store_root`、`max_rounds`

prompt text 可以直接内联在 `system_prompt` / `user_template`，也可以通过 `system_prompt_file` / `user_template_file` 从文件读取；文件路径相对于 `client/experiment.json` 解析。示例配置可从 `client/experiment.json.example` 复制。

运行方式：

```bash
python -m client.run_once
python -m client.run_loop
```

### bridge 配置

`bridge/server1/bridge.py` 会优先读取 `bridge/server1/.env`，`bridge/server2/bridge.py` 会优先读取 `bridge/server2/.env`：

- `bridge/server1/bridge.py`
  - `REMOTE_WSS_URL`
  - `REMOTE_ORIGIN`
  - `REMOTE_REFERER`
  - `LOCAL_HOST`
  - `LOCAL_PORT`
  - `MODEL_NAME`
  - `COOKIE_FILE`
- `bridge/server2/bridge.py`
  - `REMOTE_BASE_URL`
  - `REMOTE_ORIGIN`
  - `REMOTE_REFERER`
  - `LOCAL_HOST`
  - `LOCAL_PORT`
  - `COOKIE_FILE`
  - `REQUEST_TIMEOUT_SECONDS`

示例：

```bash
cat > bridge/server1/.env <<'EOF'
REMOTE_WSS_URL=wss://your-server1-proxy/ws
REMOTE_ORIGIN=https://your-server1-origin
REMOTE_REFERER=https://your-server1-referer
LOCAL_HOST=127.0.0.1
LOCAL_PORT=18000
MODEL_NAME=Qwen/Qwen3.5-35B-A3B
COOKIE_FILE=cookie.txt
EOF
python bridge/server1/bridge.py
```

```bash
cat > bridge/server2/.env <<'EOF'
REMOTE_BASE_URL=https://your-server2-proxy
REMOTE_ORIGIN=https://your-server2-origin
REMOTE_REFERER=https://your-server2-referer
LOCAL_HOST=127.0.0.1
LOCAL_PORT=19000
COOKIE_FILE=cookie.txt
REQUEST_TIMEOUT_SECONDS=30
EOF
python bridge/server2/bridge.py
```

`server1/server_ws_proxy.py` 会优先读取 `server1/.env`。  
`server2/eval_service.py` 也会优先读取 `server2/.env`，这样你后续如果给 `server2` 加评测配置，不需要再改代码。

## Topology

- The local machine runs the controller logic from `client/controller.py`.
- `server1` is the remote LLM brain exposed as an OpenAI-compatible `POST /v1/chat/completions` endpoint.
- `server2` is the remote evaluator exposed over ordinary HTTP with `POST /eval/jobs` and `GET /eval/jobs/{job_id}`.
- The local bridge layer is split into:
  - `bridge/server1/bridge.py`
  - `bridge/server2/bridge.py`
- Each local bridge uses its own `cookie.txt`.
- The controller only talks to local bridge URLs such as `http://127.0.0.1:18000` and `http://127.0.0.1:19000`.

## Label Mapping

- `A` means normal.
- `B` through `J` mean attack.
- Any output outside `A-J` counts as normal.

## Startup Order

1. Start `server1`.
   ```bash
   cat > server1/.env <<'EOF'
   VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
   EOF
   uvicorn server1.server_ws_proxy:app --host 0.0.0.0 --port 8000
   ```
2. Start `server2`.
   ```bash
   uvicorn server2.eval_service:app --host 0.0.0.0 --port 19000
   ```
3. Start the local `bridge/server1/bridge.py` process.
   ```bash
   cat > bridge/server1/.env <<'EOF'
   REMOTE_WSS_URL=wss://<server1-host>/v1/chat/completions
   REMOTE_ORIGIN=https://<server1-origin>
   REMOTE_REFERER=https://<server1-referer>
   LOCAL_HOST=127.0.0.1
   LOCAL_PORT=18000
   MODEL_NAME=Qwen/Qwen3.5-35B-A3B
   COOKIE_FILE=cookie.txt
   EOF
   python bridge/server1/bridge.py
   ```
4. Start the local `bridge/server2/bridge.py` process.
   ```bash
   cat > bridge/server2/.env <<'EOF'
   REMOTE_BASE_URL=https://<server2-host>/proxy/19000
   REMOTE_ORIGIN=https://<server2-origin>
   REMOTE_REFERER=https://<server2-referer>
   LOCAL_HOST=127.0.0.1
   LOCAL_PORT=19000
   COOKIE_FILE=cookie.txt
   REQUEST_TIMEOUT_SECONDS=30
   EOF
   python bridge/server2/bridge.py
   ```
5. Run the local client runner on your machine.

   ```bash
   cat > client/.env <<'EOF'
   SERVER1_BASE_URL=http://127.0.0.1:18000
   SERVER2_BASE_URL=http://127.0.0.1:19000
   POLL_INTERVAL_SECONDS=2.0
   EOF
   ```

`client/.env` only controls runtime URLs and polling. The experiment configuration lives in `client/experiment.json`:

- `baseline_candidate` owns the baseline candidate, prompt text, and `candidate_id`
- `best_metrics` sets the initial best metrics
- `metric_config` defines the metric policy and dataset paths: `primary_metric`, `min_value`, `tp_path`, and `tn_path`
- `runner` sets `brain_model`, `store_root`, and `max_rounds`

Prompt text can be inline in `system_prompt` / `user_template` or file-backed with `system_prompt_file` / `user_template_file`. File paths resolve relative to `client/experiment.json`. Copy `client/experiment.json.example` to get started.

Run either client entrypoint with:

```bash
python -m client.run_once
python -m client.run_loop
```

## Bridge Configuration

`bridge/server1/bridge.py` reads its settings from `bridge/server1/.env` first, then falls back to process environment variables. `bridge/server2/bridge.py` does the same with `bridge/server2/.env`:

- `bridge/server1/bridge.py`
  - `REMOTE_WSS_URL`
  - `REMOTE_ORIGIN`
  - `REMOTE_REFERER`
  - `LOCAL_HOST`
  - `LOCAL_PORT`
  - `MODEL_NAME`
  - `COOKIE_FILE`
- `bridge/server2/bridge.py`
  - `REMOTE_BASE_URL`
  - `REMOTE_ORIGIN`
  - `REMOTE_REFERER`
  - `LOCAL_HOST`
  - `LOCAL_PORT`
  - `COOKIE_FILE`
  - `REQUEST_TIMEOUT_SECONDS`

Example:

```bash
export REMOTE_WSS_URL="wss://your-server1-proxy/ws"
export REMOTE_ORIGIN="https://your-server1-origin"
export REMOTE_REFERER="https://your-server1-referer"
export MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
export COOKIE_FILE="cookie.txt"
python bridge/server1/bridge.py
```

```bash
export REMOTE_BASE_URL="https://your-server2-proxy"
export REMOTE_ORIGIN="https://your-server2-origin"
export REMOTE_REFERER="https://your-server2-referer"
export COOKIE_FILE="cookie.txt"
python bridge/server2/bridge.py
```
