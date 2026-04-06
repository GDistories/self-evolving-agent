# Server2 Bridge Design

Date: 2026-04-06

## Goal

Add a dedicated local bridge for `server2` so the local machine talks to both remote services through local bridge processes.

After this change:

- `server1` remains reachable through a local bridge that handles remote WebSocket access
- `server2` becomes reachable through a local bridge that handles remote HTTP access
- the local controller only talks to local base URLs and no longer needs to know any remote cookie or proxy details

## Problem

The current structure is asymmetric:

- `server1` already uses a local bridge because its remote access path depends on cookie-authenticated proxy access and long-running streaming responses
- `server2` is currently modeled as a directly reachable HTTP service

In reality, `server2` also sits behind the same style of cookie-authenticated reverse proxy path and must be accessed through a second cookie and proxy route such as:

- `https://.../proxy/19000/eval/jobs`

That means the local controller should not call `server2` directly. It should call a local `server2 bridge`, just like it calls a local `server1 bridge`.

## Scope

This design covers only the communication layer change.

It does not change:

- prompt evolution logic
- judge logic
- evaluator semantics
- the label mapping rule

It only changes:

- bridge layout
- local-to-remote transport for `server2`
- configuration layout
- local client integration points

## Directory Layout

The bridge layer should be split into two independent subdirectories:

- `bridge/server1/`
- `bridge/server2/`

Each subdirectory should contain:

- `bridge.py`
- `.env`
- `.env_example`
- `cookie.txt`

This keeps the two authentication domains separate and avoids coupling their cookies, proxy URLs, and protocol details.

## Responsibilities

### `bridge/server1/bridge.py`

This remains the local bridge for the remote LLM endpoint.

Responsibilities:

- read `server1` bridge config from `bridge/server1/.env`
- read `server1` cookie from `bridge/server1/cookie.txt`
- connect to the remote WSS proxy endpoint
- expose a local OpenAI-compatible `/v1/chat/completions` interface

### `bridge/server2/bridge.py`

This is the new local bridge for the remote evaluator endpoint.

Responsibilities:

- read `server2` bridge config from `bridge/server2/.env`
- read `server2` cookie from `bridge/server2/cookie.txt`
- forward local HTTP requests to the remote proxy-backed evaluator URL
- expose the same local routes as the remote evaluator:
  - `POST /eval/jobs`
  - `GET /eval/jobs/{job_id}`

This bridge is HTTP-only and does not need WebSocket or token streaming support.

## Local Client Contract

After this change, the local controller should only know about local bridge addresses:

- `SERVER1_BASE_URL=http://127.0.0.1:<server1-bridge-port>`
- `SERVER2_BASE_URL=http://127.0.0.1:<server2-bridge-port>`

The local client code must not know:

- remote proxy URLs
- cookies
- Jupyter reverse-proxy paths
- remote `Origin` / `Referer` values

All of that belongs inside the bridge layer.

## Server2 Bridge Request Flow

For a local request:

- `POST http://127.0.0.1:<port>/eval/jobs`
- `GET http://127.0.0.1:<port>/eval/jobs/{job_id}`

the `server2 bridge` should:

1. load its cookie from `cookie.txt`
2. attach:
   - `Cookie`
   - `Origin`
   - `Referer`
3. forward the request to:
   - `https://.../proxy/19000/eval/jobs`
   - `https://.../proxy/19000/eval/jobs/{job_id}`
4. return the upstream status code and body to the local caller

The bridge should behave as a thin authenticated reverse proxy.

## Configuration

### `bridge/server1/.env`

Expected settings:

- `REMOTE_WSS_URL`
- `REMOTE_ORIGIN`
- `REMOTE_REFERER`
- `LOCAL_HOST`
- `LOCAL_PORT`
- `MODEL_NAME`

### `bridge/server2/.env`

Expected settings:

- `REMOTE_BASE_URL`
- `REMOTE_ORIGIN`
- `REMOTE_REFERER`
- `LOCAL_HOST`
- `LOCAL_PORT`
- `COOKIE_FILE`
- `REQUEST_TIMEOUT_SECONDS`

`REMOTE_BASE_URL` should be something like:

- `https://.../proxy/19000`

so the bridge can append `/eval/jobs` and `/eval/jobs/{job_id}` locally and forward them directly.

## Error Handling

The `server2 bridge` must make transport problems explicit.

Minimum required handling:

- missing `cookie.txt`
  - fail fast with a clear local error response
- upstream non-200 response
  - return upstream status code and body
- upstream timeout
  - return a recognizable timeout error
- upstream non-JSON body
  - return text body rather than hiding it

The goal is to make reverse-proxy and authentication issues debuggable from the local machine.

## Testing

This bridge change should be tested in three layers.

### Unit tests

Cover:

- `.env` loading
- URL join behavior for remote `server2` forwarding
- auth/header construction

### Bridge API tests

Mock the remote proxy URL and verify:

- local `POST /eval/jobs` forwards correctly
- local `GET /eval/jobs/{job_id}` forwards correctly
- status code and body are preserved

### Client regression tests

Verify the local `EvaluatorClient` does not need interface changes. It should only need a different `SERVER2_BASE_URL`.

## Non-Goals

This change does not:

- alter the evaluator API shape
- add new prompt evolution logic
- change `A/B-J` mapping behavior
- introduce persistent bridge state

## Expected Outcome

After this change:

- `server1` and `server2` are accessed through symmetric local bridge processes
- each bridge owns its own cookie and remote proxy configuration
- the local controller only depends on local URLs
- remote access details are fully isolated inside `bridge/server1/` and `bridge/server2/`
