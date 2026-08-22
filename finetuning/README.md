# Ariadne fine-tuning pipeline

QLoRA supervised fine-tuning for one narrow task: **natural-language request →
Ariadne `AnalysisPlan`**.

This directory is not part of the web service. It has its own `pyproject.toml`,
is never imported by `app/`, and is never started by FastAPI. The only thing it
shares with Ariadne is a versioned dataset directory produced by the exporter.

```
Ariadne DB → approved candidate exporter → data/analysis_plan_vN.jsonl → this pipeline
```

## What is deliberately out of scope

Version 1 does not fine-tune "the agent". It trains planning and
structured-output reliability only, excluding Overpass execution, Nominatim
resolution, coordinate generation, GeoJSON, GIS calculations, hidden reasoning
and multi-turn tool orchestration. Those stay with the deterministic backend,
which is where they belong.

DPO is not implemented. It becomes interesting once the review queue contains
enough approved/rejected pairs *for the same request*; today it does not.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | Experiment configuration (YAML) |
| `data/` | Exported datasets (gitignored) |
| `src/ariadne_finetuning/` | Library code |
| `scripts/` | Command-line entry points |
| `evaluation/` | Gold evaluation set and metric documentation |
| `tests/` | Offline tests (no GPU, no model download) |
| `artifacts/` | Adapters, reports, registry (gitignored) |

Modules:

| Module | Responsibility |
|---|---|
| `config.py` | Dataclasses, YAML/JSON loading, `BASE_MODEL_ID`-style env overrides |
| `dataset.py` | Load and validate exported splits; leakage report; frozen test digest |
| `schema.py` | Dependency-free checking against the exported plan JSON Schema |
| `metrics.py` | Per-dimension plan metrics |
| `evaluation.py` | Frozen-test-set evaluation and before/after comparison |
| `modeling.py` | 4-bit loading, LoRA config, generation (lazy ML imports) |
| `training.py` | TRL `SFTTrainer` wiring and adapter metadata |
| `pipeline.py` | The whole sequence, with injectable trainer and runners |
| `registry.py` | Local experiment metadata and promotion status |

Every machine-learning import lives inside a function, so the library imports,
type-checks and tests without torch, a GPU or a download.

## Install

```bash
# library only: validation, metrics, evaluation, registry, pipeline wiring
pip install -e .

# plus the training stack (GPU box or Colab)
pip install -e '.[train]'
```

## Workflow

```bash
# 1. Export approved candidates from Ariadne (repository root)
./.venv/bin/python -m app.cli export-training-dataset --dataset-version v1

# 2. Validate before spending GPU time
python scripts/validate_dataset.py --config configs/qlora_analysis_plan.yaml

# 3. Baseline the untrained model on the frozen test split
python scripts/train_qlora.py --config configs/qlora_analysis_plan.yaml --baseline-only

# 4. Train, then evaluate on exactly the same examples
python scripts/train_qlora.py --config configs/qlora_analysis_plan.yaml

# 5. Record a human decision (this changes nothing in production)
python scripts/promote.py --experiment-id analysis-plan-qlora-001 --status approved \
    --notes "improves target preservation; reviewed by <name>"
```

The pipeline is: dataset → validation → split check → base model → 4-bit
quantization → `prepare_model_for_kbit_training` → LoRA config → `SFTTrainer` →
train → save adapter → evaluate → report. It stops there.

## Configuration

`configs/qlora_analysis_plan.yaml` holds the defaults. Any of these environment
variables override it: `BASE_MODEL_ID`, `EXPERIMENT_ID`, `DATASET_DIR`,
`DATASET_VERSION`, `OUTPUT_DIR`, `MAX_SEQ_LENGTH`, `LEARNING_RATE`,
`NUM_EPOCHS`, `TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LORA_R`,
`LORA_ALPHA`, `LORA_DROPOUT`.

QLoRA defaults: 4-bit NF4 with double quantization, LoRA `r=16` / `alpha=32` on
`q_proj,k_proj,v_proj,o_proj`, batch size 1 with 8-step accumulation, gradient
checkpointing on, sequence length 1024, greedy decoding at evaluation time.

These are conservative starting points, **not** a hardware compatibility claim.
Whether a given base model fits depends on the model, the sequence length and
the card. Measure it.

## Artifacts

Training writes a PEFT/LoRA adapter (not a merged checkpoint) to
`artifacts/<experiment_id>/adapter/`, alongside
`ariadne_adapter_metadata.json`: base model id, dataset version, test digest,
training example count and the full training configuration. The experiment
registry at `artifacts/registry.json` adds baseline metrics, fine-tuned
metrics, git commit and promotion status. None of it is committed.

## Promotion

`train → evaluate → manual approval → optional merge/export → optional serving
→ A/B evaluation`.

Every run is recorded as `experimental`. Nothing promotes itself, and this
pipeline never touches Ariadne's runtime model configuration.

## Future: serving through Ollama

Not implemented, and not worth implementing before an adapter is worth serving.
The path would be: Hugging Face base + LoRA adapter → `merge_and_unload` →
GGUF conversion → quantize → `ollama create` from a Modelfile → point
`OLLAMA_MODEL` at the new tag. The milestone this pipeline currently ends at is
a validated adapter plus an evaluation report.

See also [COLAB.md](COLAB.md) and
[docs/active-learning-and-finetuning.md](../docs/active-learning-and-finetuning.md).
