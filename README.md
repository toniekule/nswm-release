# NegativeSpace-WM

Minimal repair certificates, shared Critic/Gate learning, conservative verification and selective generation.

The core runs on the Python standard library. CPU corpus construction uses MuJoCo, NumPy and Pillow. Training uses the pinned training extra. Model weights and datasets are obtained or constructed separately.

## Install

Python 3.10 or newer.

```bash
python -m pip install -e '.[train,physics]'   # training and MuJoCo corpus construction
python -m pip install -e .                    # core only; heavy dependencies load on use
```

Verify the installation:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/check_code.py                  # contracts and in-memory checks, no training
python -m unittest discover -s tests -v       # full suite, includes tiny training runs
```

`scripts/cpu_smoke.sh` runs the full suite plus the planning demonstration.

## Quick start without weights

`make-tiny` writes random Qwen3-VL weights, a local video processor and tokenizer, a fixture
manifest, and a CPU training configuration. The ordinary commands then run end to end.

```bash
nswm make-tiny .runs/tiny --seed 0
nswm resources .runs/tiny/train.json
nswm train .runs/tiny/train.json
nswm evaluate .runs/tiny/data/events.jsonl .runs/tiny/training/best.pt .runs/tiny/evaluation --split dev --modes gate --arm tiny
nswm report .runs/tiny/evaluation/predictions.jsonl --out .runs/tiny/report.json
```

Every mode uses real video tensors. Output commands refuse an existing destination.

## Build a corpus

```bash
nswm corpus --spec configs/mechanisms.json --split-spec configs/splits.json --out .runs/mechanisms --families 240 --seed 0
nswm audit-data .runs/mechanisms/corpus-train/events.jsonl
nswm audit-data .runs/mechanisms/eval-ood/events.jsonl
nswm train configs/train-mechanisms.json --plan
```

Add `--plan` to the first command to validate quotas and print the scene and split declarations
without simulating, rendering or writing files. With a split specification the family count must
be a multiple of 240; without one it must be a multiple of 60. Use `configs/scenes.json` for the
smaller sphere corpus (`--families 100`).

Each population contains `events.jsonl`, PNG observations, scene XML, replay evidence,
`data_card.json` and `isolation.json`. The data card records counts, split rules, seeds, dependency
versions and artifact hashes. Builds whose isolation audit fails are left marked incomplete.

`--renderer mujoco` selects MuJoCo offscreen rendering instead of the default CPU projection; set
`MUJOCO_GL=osmesa` or `egl` beforehand on those Linux backends.

## Robot assets

```bash
nswm register-robot --assets assets/robot-model --xml scene.xml --repository "$NSWM_ROBOT_REPOSITORY" --revision "$NSWM_ROBOT_REVISION" --licenses robot-licenses.json --out assets/robot-receipt.json
nswm robot-check --assets assets/robot-model --receipt assets/robot-receipt.json --spec robot-task.json
```

Registration hashes local model files and their licenses; it never downloads assets. The license
file is a JSON array of `{model, spdx, file}` entries. XML includes and mesh or texture references
must stay inside the registered directory. A task spec holds version 1, initial `qpos`/`qvel`,
action arrays, position assertions (`slot`, `body`, `frame`, `axis`) and goals
(`body`, `frame`, `axis`, `interval`). `robot-check` runs a fixed-action CPU replay and checks the
declared goals.

## Obtain and convert a base

```bash
nswm fetch --repo nvidia/Cosmos3-Nano --revision "$NSWM_COSMOS_REVISION" --role cosmos --out assets/Cosmos3-Nano
nswm fetch --repo Qwen/Qwen3-VL-8B-Instruct --revision "$NSWM_PROCESSOR_REVISION" --role processor --out assets/Qwen3-VL-8B-Instruct
nswm convert-base --source assets/Cosmos3-Nano --processor assets/Qwen3-VL-8B-Instruct --out assets/Cosmos3-Nano-VLM
nswm resources configs/train-mechanisms.json --probe-device
nswm train configs/train-mechanisms.json
```

`fetch` is the only command that downloads weights and requires a full immutable Hub commit hash.
It records the repository, revision and file hashes; processor acquisition excludes weight files.
Authentication uses the standard Hugging Face environment or local login.

`convert-base` reads local safetensors from the understanding and vision towers and writes a
complete Qwen3-VL directory. Missing, duplicate or incompatible tensors fail rather than falling
back to other weights, and output shards are streamed. The receipt records source hashes, tensor
mappings and converted identities. Use `--shard-mib` to set the output shard size.

`resources` reports weight headers, processor components, hashes, training dependencies and
manifest availability without importing Torch. Add `--probe-device` to check CUDA.

## Train and resume

```bash
nswm train configs/train-mechanisms.json
nswm train configs/train-mechanisms.json --resume checkpoints/mechanisms-shared-cert-seed0/latest.pt
nswm evaluate .runs/mechanisms/eval-ood/events.jsonl checkpoints/mechanisms-shared-cert-seed0/best.pt .runs/evaluation --arm certificate --seed 0
```

Paths in a configuration are interpreted from the invocation directory, and training loads local
files only.

| Configuration | Behavior |
|---|---|
| `events`, `model`, `processor`, `output` | Manifest, base weights, processor and run directory (required) |
| `modes` | Shared `["gate","critic"]` or Gate-only `["gate"]` |
| `supervision` | Certificate, redundant, binary, scalar, free-form or tied field queries; `minimal` is rejected |
| `removed_fields`, `restricted_critic` | Target-field masking and history-only Critic input |
| `effective_batch`, `epochs`, `seed` | Deterministic family exposure and mode alternation |
| `population_subset`, `population_modes` | Eligible family population (`all`, `certificate`, `four_target`) and its frozen scope |
| `rank`, `alpha`, `dropout` | Language attention adapters; defaults 64 / 128 / 0.05 |
| `max_steps` | Stop after this many updates without changing the full schedule |
| `validation_interval` | Defaults to `max(8, total_updates // 4)`; the last completed update is also validated |
| `validation_families` | Optional deterministic development-family limit |
| `max_text_tokens`, `max_output_tokens` | Strict context and target limits; overlong assertions are rejected |
| `decode_max_tokens` | Optional inference-only generation limit, otherwise the output limit |
| `longest_side`, `fps`, `gradient_checkpointing` | Frame sizing, timestamps and memory use |

`latest.pt` holds the adapter, optimizer, RNG state and identity binding; resuming against a
changed base, processor, manifest, media or configuration is rejected. `best.pt` is selected by
development loss, including at the end of a short run, and no checkpoint is selected when the
development population is empty. `training.jsonl` and `validation.jsonl` record per-update
readings and family IDs. Checkpoints load with `weights_only=True`.

## Evaluate and report

```bash
nswm report .runs/evaluation/predictions.jsonl --out .runs/evaluation/report.json
nswm report runs/arm-a/seed-*/predictions.jsonl runs/arm-b/seed-*/predictions.jsonl --compare arm-a arm-b --out runs/comparison.json
```

Raw records preserve mode, arm, seed, family, domain, OOD axis, query and event bindings, the
decoded certificate, risk and the compute unit. Reports group by arm and mode and include
balanced accuracy, field exact match, `J_valid`, `J_cert`, repair success, conditional minimality,
unknown counts and compute totals per unit, with slices over domain, OOD axis, category,
difficulty and variant. Paired contrasts require identical evaluation populations and at least two
identified training seeds.

Certificate-free inventory diagnostics (feasible rejection, infeasible retention, resolution) are
reported separately under `inventory_diagnostics` and never enter model-arm tables or contrasts.
Physical quality metrics require a matching assessor; manifests without one are still recognised
and serialised, but those metrics stay unavailable.

## Planning and optional backends

```bash
nswm plan-demo --policy certificate --budget 10
```

Eight planner policies share proposal, dispatch, acceptance, scoring, execution and replanning with
an episode cost ledger. The Certificate policy cancels only on an independently proven conflict
whose scope covers the complete request; unresolved or insufficiently covered requests are
retained. The demonstration reports deterministic work units.

Optional adapters provide SAM 2 history segmentation, online CoTracker3 tracking, an injected
FoundationPose estimator, nominal FCL queries, JSON process generation and Cosmos
action-conditioned generation. Their checkpoints, embodiment data and hardware are provisioned
separately.

## Source release

This repository is the release tree. To build an archive with a per-file checksum manifest:

```bash
python scripts/package_release.py .runs/release/nswm.tar.gz
```

The archive contains package source, instructions, configurations, examples, scripts, tests and CI.
It excludes data, weights, caches, checkpoints and local notes.

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE). Dependencies and externally
referenced components keep their own terms: PyTorch and TorchVision (BSD-3-Clause), Transformers,
Accelerate, safetensors and MuJoCo (Apache-2.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU), Qwen3-VL
(Apache-2.0), SAM 2 (Apache-2.0), CoTracker3 (CC-BY-NC-4.0, non-commercial), python-fcl
(BSD-3-Clause) and IBEX (LGPL-3.0). No third-party source is vendored, and no weights or datasets
are distributed.
