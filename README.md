# self-evolving-agent

## Topology

- The local machine runs the controller logic from `client/controller.py`.
- `server1` is the remote LLM brain exposed as an OpenAI-compatible `POST /v1/chat/completions` endpoint.
- `server2` is the remote evaluator exposed over ordinary HTTP with `POST /eval/jobs` and `GET /eval/jobs/{job_id}`.

## Label Mapping

- `A` means normal.
- `B` through `J` mean attack.
- Any output outside `A-J` counts as normal.

## Startup Order

1. Start the `server1` bridge that fronts your remote model endpoint. In this repo that bridge is `server/server_ws_proxy.py`.
   ```bash
   export VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
   uvicorn server.server_ws_proxy:app --host 0.0.0.0 --port 8000
   ```
2. Start the evaluator service on the remote side.
   ```bash
   uvicorn server2.eval_service:app --host 0.0.0.0 --port 19000
   ```
3. Run the local controller process or driver on your machine, pointing it at the two base URLs with `SERVER1_BASE_URL` and `SERVER2_BASE_URL`.

The controller itself is library code today, so the entrypoint is whatever local runner imports `run_iteration` from `client/controller.py`.

## Bridge Configuration

`client/client_ws_proxy.py` now reads its remote bridge settings from environment variables:

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
python client/client_ws_proxy.py
```
