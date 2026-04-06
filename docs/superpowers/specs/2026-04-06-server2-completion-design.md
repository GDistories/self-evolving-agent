# Server2 Completion Design

`server2` keeps the current HTTP contract:

- `POST /eval/jobs`
- `GET /eval/jobs/{job_id}`

The request payload continues to come from the local client, but `tp_dataset` and `tn_dataset` are interpreted as paths on the `server2` machine. `server2` reads those JSONL files locally, evaluates them with the 8B model, and returns stable structured results.

Label handling follows the existing project rule:

- `A` means normal
- `B` through `J` mean attack
- any output outside `A-J` is treated as normal

Evaluation returns:

- `tp_stats`
- `tn_stats`
- `merged_metrics`
- `failure_samples`
- `raw_artifacts`

`server2` supports two model lifecycle modes:

- `per_job`: load model for each job, release after completion
- `lazy_reuse`: load model on first job, reuse for later jobs

Caching is based on semantic inputs, not only file paths. The cache key includes prompt content, dataset file content fingerprints, runtime model configuration, and inference configuration. Background execution is asynchronous inside the `server2` process and job statuses move through `queued`/`running`/`completed` or `failed`, with immediate `cached` on cache hit.
