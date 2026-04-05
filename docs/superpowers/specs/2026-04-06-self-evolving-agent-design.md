# Self-Evolving Agent Design

Date: 2026-04-06

## Goal

Build a locally run self-evolving agent that improves prompt-based prompt-attack detection through repeated evaluation and revision.

The local program is the only controller. It has no large model locally. Instead:

- `server1` provides the remote large-model brain through `/v1/chat/completions`
- `server2` provides remote evaluation using an 8B model and returns structured results

The first implementation must support:

- iterative optimization of `system_prompt`
- iterative optimization of a user-side wrapper template
- dynamic evaluation metrics provided at runtime
- asynchronous evaluation jobs on `server2`
- result caching on `server2`
- a single best-candidate optimization loop on the local machine

The first implementation must not depend on:

- hard-coded metric thresholds in code
- multi-branch search trees
- complex strategy learning on `server2`
- local large-model inference

## Scope

This design covers one end-to-end optimization loop:

1. Start from a baseline prompt configuration
2. Ask `server1` to generate a new candidate
3. Submit the candidate to `server2`
4. Evaluate against `tp.jsonl` and `tn.jsonl`
5. Compare results against dynamic runtime criteria
6. Promote, reject, or continue iterating

The initial dataset format is fixed to two JSONL files:

- `tp.jsonl`
- `tn.jsonl`

Each record uses:

```json
{"id": "...", "text": "..."}
```

The label is implied by the file identity rather than stored per record.

## Architecture

The system has three runtime roles.

### 1. Local Agent Controller

The local controller orchestrates the full loop. It is responsible for:

- loading the baseline candidate
- loading dynamic evaluation requirements for the run
- calling `server1` to propose a new candidate
- submitting jobs to `server2`
- polling job status
- comparing results with the current best candidate
- deciding whether to stop, continue, or promote
- storing experiment history locally

This component owns all optimization policy and all persistent local experiment records.

### 2. Server1: Candidate Generator

`server1` is the remote large-model endpoint used as the agent brain. It only generates candidate updates. It does not store loop state and does not judge quality directly.

Responsibilities:

- receive structured context from the local controller
- return a revised candidate
- explain intended changes in a machine-readable or parseable form

### 3. Server2: Evaluator

`server2` is the remote evaluation service backed by the 8B model. It does not decide optimization strategy. It only executes evaluation jobs, caches results, and returns structured outputs.

Responsibilities:

- accept candidate evaluation jobs
- execute evaluation asynchronously
- cache results for repeated equivalent jobs
- return merged metrics and failure samples

## Candidate Model

A candidate is the atomic unit of optimization. It represents one version under test.

Suggested fields:

- `candidate_id`
- `parent_candidate_id`
- `system_prompt`
- `user_template`
- `mutation_note`
- `created_at`
- `created_by`
- `status`

Status values should support at least:

- `draft`
- `testing`
- `passed`
- `rejected`
- `best`

The first version only needs to mutate:

- `system_prompt`
- `user_template`

The interface should already leave room for future extension to generation parameters such as `temperature`, `max_tokens`, and constrained output settings.

## Evaluation Spec

Evaluation criteria must be supplied at runtime, not hard-coded.

Suggested fields:

- `tp_path`
- `tn_path`
- `metric_config`
- `max_rounds`
- `top_k_failures_to_feedback`
- `stop_conditions`

`metric_config` is intentionally externalized because the target indicator will change over time.

The first implementation should support deterministic local judging from structured config rather than embedding metric policy inside model prompts.

## Evaluation Result

The evaluation result returned from `server2` should be stable and structured.

Suggested fields:

- `job_id`
- `candidate_id`
- `dataset_fingerprint`
- `cache_hit`
- `tp_stats`
- `tn_stats`
- `merged_metrics`
- `failure_samples`
- `raw_artifacts`

`failure_samples` should distinguish at least:

- missed positive cases from `tp.jsonl`
- false alarms from `tn.jsonl`

## Server2 API

The initial service contract should expose asynchronous job semantics.

### `POST /eval/jobs`

Submit a new evaluation job.

Input should include:

- `candidate_id`
- `system_prompt`
- `user_template`
- `tp_dataset`
- `tn_dataset`
- `metric_config`

Response should include:

- `job_id`
- `status`
- optional summary when the cache is hit immediately

Status values should support:

- `queued`
- `running`
- `cached`
- `completed`
- `failed`

### `GET /eval/jobs/{job_id}`

Return current state and final result when available.

Response should include:

- `status`
- `progress`
- `result` or `error`

### Cache lookup boundary

The first implementation does not need a separate cache lookup endpoint. Cache resolution should happen inside `POST /eval/jobs`.

A dedicated cache lookup API may be added later only if operational needs justify it.

## Cache Design

Result caching is required in the first implementation.

The cache key must hash all semantically relevant evaluation inputs:

- `system_prompt`
- `user_template`
- `tp.jsonl` content hash
- `tn.jsonl` content hash
- evaluator model version on `server2`
- relevant inference settings

This prevents false cache hits when prompt content, datasets, or evaluation runtime change.

## Optimization Loop

The local controller runs a bounded iterative loop.

### Loop steps

1. Load the current best candidate
2. Send the best candidate, recent history, metrics, and representative failures to `server1`
3. Receive a revised candidate from `server1`
4. Submit the candidate to `server2`
5. Poll until the job completes
6. Judge the result locally against the current best and runtime metric policy
7. Promote the candidate if it wins
8. Continue until stop conditions trigger

### Feedback sent to `server1`

Each iteration should include:

- the current best candidate
- recent mutation history for the last few rounds
- summary metrics from the latest evaluation
- representative `tp` failures
- representative `tn` failures

The local controller should limit failure samples to a small, curated subset. Sending every failure back to `server1` would create noisy and unstable revisions.

### Candidate output required from `server1`

The local controller should request at least:

- revised `system_prompt`
- revised `user_template`
- concise mutation rationale
- expected impact of the change

This rationale is required for traceability and later debugging.

## Judge

The promotion decision must be deterministic and local.

`server1` may propose changes, but it must not decide whether a candidate succeeds. The local controller owns that decision through a `Judge` component.

The `Judge` should support:

- checking hard constraints from `metric_config`
- comparing a challenger against the current best
- returning a clear decision and reason

The first implementation should use a simple single-best strategy:

- one `best` candidate
- one current challenger under test

This avoids unnecessary complexity while still enabling full closed-loop optimization.

## Stop Conditions

The loop stops when any configured stop condition is met.

The first implementation should support at least:

- target metrics satisfied
- maximum rounds reached
- no meaningful improvement for N rounds
- repeated or near-duplicate revisions from `server1`
- repeated cache hits without effective candidate changes

## Error Handling

The first implementation must explicitly handle failure modes across the three-part chain.

### Server1 failures

- bounded retry on transient request failure
- do not overwrite the current best on generation failure
- persist the failed attempt record locally

### Server2 failures

- mark the job as failed
- preserve submitted inputs for replay
- allow the local controller to retry or skip

### Invalid results

- reject malformed result payloads
- reject result sets with inconsistent sample counts
- prevent invalid runs from entering the judge step

### Cache safety

- never cache by `candidate_id` alone
- require content-derived fingerprints

## Testing Strategy

Testing should be staged.

### Unit tests

Cover:

- judge decisions
- cache fingerprint generation
- failure-sample selection
- result aggregation and schema validation

### Integration tests

Mock `server1` and `server2` to verify:

- loop progression
- promotion and rejection logic
- retry behavior
- stop-condition behavior

### Smoke test

Run a small real dataset on:

- a baseline candidate
- one or two evolved candidates

This smoke test should verify:

- job submission
- polling
- result caching
- feedback shaping
- local experiment persistence

## Implementation Order

The first implementation should proceed in this order:

1. convert `server2/batch_run.py` into an evaluation service core
2. add asynchronous job handling and caching on `server2`
3. build the local controller that can submit one evaluation and read one result
4. connect the local controller to `server1`
5. implement the deterministic judge and bounded multi-round loop
6. extend the candidate schema for future parameter search without enabling it yet

## Out of Scope for First Version

The following are explicitly out of scope:

- multiple parallel candidate branches
- search-tree optimization
- automatic strategy generation on `server2`
- learned reward models
- local large-model deployment
- broad hyperparameter search beyond prompt and template editing

## Expected Outcome

After the first version, the project should be able to:

- run locally as a full optimization controller
- use `server1` as the remote large-model brain
- use `server2` as an asynchronous cached evaluation backend
- improve prompt-based attack detection through repeated evidence-driven revisions
- keep an auditable record of how and why each candidate changed
