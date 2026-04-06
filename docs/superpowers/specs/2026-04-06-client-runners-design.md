# Client Runners Design

Date: 2026-04-06

## Goal

Add two local client entrypoints for the existing self-evolving-agent codebase:

- a one-shot runner for end-to-end smoke testing
- a multi-round runner for formal iterative optimization

The design must keep runtime environment wiring separate from experiment content, and it must support baseline prompts defined either inline in config or loaded from Markdown files.

## Scope

This design covers:

- the runtime shape of `client/run_once.py`
- the runtime shape of `client/run_loop.py`
- configuration boundaries for bridge URLs versus experiment inputs
- baseline prompt loading rules
- validation behavior for ambiguous or incomplete prompt config

This design does not change:

- the `server1` or `server2` API contracts
- the existing `run_iteration(...)` controller contract
- the evaluation payload schema already used between client and `server2`

## Runtime Roles

### 1. `client/run_once.py`

This runner exists only to verify that the full chain is operational:

- local client
- local `bridge/server1`
- local `bridge/server2`
- remote `server1`
- remote `server2`

It loads runtime config and experiment config, resolves the baseline candidate, constructs clients, executes exactly one `run_iteration(...)`, and prints a concise result summary.

It should still persist artifacts through the existing `ExperimentStore` path so the user can inspect the generated challenger and iteration record after the smoke test.

### 2. `client/run_loop.py`

This runner is the formal optimization entrypoint. It loads the same configuration inputs as `run_once.py` but repeats the loop until it reaches a stop condition.

The first version only needs one required stop condition:

- `max_rounds`

Each round should:

1. call `run_iteration(...)`
2. update the current best candidate and best metrics from the outcome
3. print concise per-round progress
4. stop once `max_rounds` is reached

Future stop conditions can be added later without changing the configuration boundary introduced here.

## Configuration Layers

The design uses two distinct configuration sources.

### Runtime Environment: `client/.env`

`client/.env` remains responsible only for service connectivity and polling behavior:

- `SERVER1_BASE_URL`
- `SERVER2_BASE_URL`
- `POLL_INTERVAL_SECONDS`

This file answers "where does the client connect?"

It must not become the place for baseline prompts, dataset paths, or optimization policy.

### Experiment Definition: `client/experiment.json`

`client/experiment.json` defines what the client is optimizing and how the run should behave.

Suggested top-level sections:

- `baseline_candidate`
- `best_metrics`
- `metric_config`
- `runner`

Suggested contents:

#### `baseline_candidate`

- `candidate_id`
- either `system_prompt` or `system_prompt_file`
- either `user_template` or `user_template_file`

#### `best_metrics`

The initial best-known metrics used as the comparison baseline for the first iteration.

#### `metric_config`

The same dynamic evaluation policy already expected by `run_iteration(...)`, including fields such as:

- `tp_path`
- `tn_path`
- `primary_metric`
- `min_value`

#### `runner`

- `brain_model`
- `store_root`
- `max_rounds`

`run_once.py` loads this section but ignores `max_rounds`.

## Prompt Loading Rules

Both `system_prompt` and `user_template` support two mutually exclusive input modes.

### Inline mode

The experiment config may define:

- `system_prompt`
- `user_template`

These values are used directly as prompt text.

### File mode

The experiment config may define:

- `system_prompt_file`
- `user_template_file`

The client reads the referenced file contents as plain text and uses the full contents as the prompt value.

Markdown files are supported as ordinary text files. No Markdown parsing, stripping, or rendering behavior should be added. A `.md` file is only a storage format convenience for the user.

### Path resolution

Prompt file paths should resolve relative to the directory containing `client/experiment.json`.

This keeps experiment bundles portable when moved within the repository.

## Validation Rules

Prompt source configuration must fail fast when ambiguous or incomplete.

For each field pair:

- if both inline and file variants are present, raise a configuration error
- if neither variant is present, raise a configuration error

This rule applies independently to:

- `system_prompt` / `system_prompt_file`
- `user_template` / `user_template_file`

The runner should also fail fast when:

- `candidate_id` is missing
- required `metric_config` fields are missing
- `brain_model` is missing
- `store_root` is missing
- `max_rounds` is missing or invalid for `run_loop.py`

## Data Flow

The startup flow is:

1. load `client/.env` through existing runtime config handling
2. load `client/experiment.json`
3. resolve baseline prompt text from inline or file-backed config
4. build a baseline `Candidate`
5. construct `OpenAIBrainClient` with:
   - `SERVER1_BASE_URL`
   - `runner.brain_model`
6. construct `EvaluatorClient` with `SERVER2_BASE_URL`
7. call `run_iteration(...)`
8. print summary output and rely on `ExperimentStore` for persistent records

`run_loop.py` repeats steps 7 and 8 until `max_rounds` is reached, carrying forward the latest best candidate and metrics after each round.

## Output Expectations

Both runners should print concise terminal output suitable for manual operation.

`run_once.py` should print:

- baseline candidate id
- generated challenger candidate id
- promotion decision
- returned metrics

`run_loop.py` should print:

- round number
- previous best candidate id
- challenger candidate id
- promotion decision
- current best candidate id after the round

At the end of the loop it should print the final best candidate id and final best metrics.

## Error Handling

The first version should keep error handling simple and explicit.

- Configuration errors should stop before any remote request is made.
- Missing prompt files should raise a local file error with the resolved path.
- HTTP failures from `server1` or `server2` should surface directly rather than being silently retried.
- Polling behavior remains governed by the existing evaluator client and runtime config.

This is sufficient for an operator-facing first version where transparency matters more than recovery automation.

## Testing

The implementation should add tests for:

- experiment config loading in inline prompt mode
- experiment config loading in file-backed prompt mode
- prompt config rejection when both inline and file keys are set
- prompt config rejection when neither key is set
- `run_once.py` orchestration with mocked clients
- `run_loop.py` stopping after `max_rounds`
- `run_loop.py` carrying forward promoted versus non-promoted best candidates correctly

## Recommendation

Keep the two runners as separate thin entrypoints over shared config-loading helpers.

This preserves a clean distinction between:

- smoke testing a live topology
- formal iterative optimization

It also avoids turning a single CLI into a branching `--mode` script that accumulates unrelated behavior over time.
