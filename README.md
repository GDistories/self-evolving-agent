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

`client/.env` 只控制运行时 URL，以及客户端等待 `server2` 评测 job 完成时使用的轮询间隔。实验本身放在 `client/experiment.json`：

- `baseline_candidate` 负责 baseline candidate、prompt text、`candidate_id`
- `best_metrics` 负责初始最优指标
- `metric_config` 负责 metric policy 和数据集路径，包括 `primary_metric`、`min_value`、`tp_path`、`tn_path`
- `runner` 负责 `brain_model`、`store_root`、`max_rounds`

prompt text 可以直接内联在 `system_prompt` / `user_template`，也可以通过 `system_prompt_file` / `user_template_file` 从文件读取；文件路径相对于 `client/experiment.json` 解析。

推荐的使用方式是先复制示例配置，再按你的实验改：

```bash
cp client/experiment.json.example client/experiment.json
mkdir -p client/prompts
cat > client/prompts/system.md <<'EOF'
You are a careful evaluator that follows the experiment policy.
EOF
```

然后重点改这几项：

- `baseline_candidate.candidate_id`
  - 这次实验的 baseline 版本号
- `baseline_candidate.system_prompt` 或 `baseline_candidate.system_prompt_file`
  - 二选一
- `baseline_candidate.user_template` 或 `baseline_candidate.user_template_file`
  - 二选一
- `best_metrics`
  - baseline 当前已知最优指标；第一轮比较会拿它做基线
- `metric_config.tp_path` / `metric_config.tn_path`
  - 你的正负样本 JSONL 路径
- `metric_config.primary_metric`
  - 本轮晋升时看的主指标
- `runner.brain_model`
  - 发给 `server1` 的模型名
- `runner.store_root`
  - 本地保存 candidate 和 iteration 记录的目录
- `runner.max_rounds`
  - `client.run_loop` 的最大迭代轮数

`client/experiment.json.example` 当前示例演示的是“`system_prompt` 从文件读取，`user_template` 直接内联”的混合写法。你也可以改成全内联，或者两个字段都走文件。

运行方式：

```bash
python -m client.run_once
python -m client.run_loop
```

- `python -m client.run_once`
  - 只跑一轮，适合先验证 server1、server2、两个 bridge 和配置是否通
- `python -m client.run_loop`
  - 连续跑到 `runner.max_rounds`

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

`client/.env` only controls runtime URLs and the polling interval used while the client waits for `server2` evaluation jobs to finish. The experiment configuration lives in `client/experiment.json`:

- `baseline_candidate` owns the baseline candidate, prompt text, and `candidate_id`
- `best_metrics` sets the initial best metrics
- `metric_config` defines the metric policy and dataset paths: `primary_metric`, `min_value`, `tp_path`, and `tn_path`
- `runner` sets `brain_model`, `store_root`, and `max_rounds`

Prompt text can be inline in `system_prompt` / `user_template` or file-backed with `system_prompt_file` / `user_template_file`. File paths resolve relative to `client/experiment.json`.

A practical way to start is:

```bash
cp client/experiment.json.example client/experiment.json
mkdir -p client/prompts
cat > client/prompts/system.md <<'EOF'
You are a careful evaluator that follows the experiment policy.
EOF
```

Then update the fields that usually change per run:

- `baseline_candidate.candidate_id`
  - your baseline version label
- `baseline_candidate.system_prompt` or `baseline_candidate.system_prompt_file`
  - choose one
- `baseline_candidate.user_template` or `baseline_candidate.user_template_file`
  - choose one
- `best_metrics`
  - the current baseline metrics used as the comparison starting point
- `metric_config.tp_path` / `metric_config.tn_path`
  - positive and negative dataset JSONL paths
- `metric_config.primary_metric`
  - the metric that decides promotion
- `runner.brain_model`
  - model name sent to `server1`
- `runner.store_root`
  - local directory for saved candidates and iteration records
- `runner.max_rounds`
  - maximum loop count for `client.run_loop`

`client/experiment.json.example` currently demonstrates a mixed setup: `system_prompt` loaded from a file and `user_template` kept inline. You can switch to fully inline or fully file-backed prompts if you prefer.

Run either client entrypoint with:

```bash
python -m client.run_once
python -m client.run_loop
```

- `python -m client.run_once`
  - runs one iteration and is useful for smoke-testing the full topology
- `python -m client.run_loop`
  - keeps iterating until `runner.max_rounds` is reached

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
