# Running the pipeline on Google Colab

Nothing in the repository changes for Colab. You upload two things — the
`finetuning/` directory and one exported dataset version — and run the same
scripts.

## 0. Check that you actually have a GPU

Colab does not always hand one out, and QLoRA on CPU is not practical.

```python
!nvidia-smi || echo "NO GPU: switch Runtime > Change runtime type > GPU"
```

```python
import torch
assert torch.cuda.is_available(), "no CUDA device; stop here rather than waiting hours"
print(torch.cuda.get_device_properties(0))
```

The pipeline exposes the same probe:

```python
from ariadne_finetuning.modeling import describe_environment
describe_environment()
```

## 1. Dependencies

```python
!pip install -q -U "transformers>=4.44" "datasets>=2.20" "peft>=0.12" \
    "trl>=0.11" "bitsandbytes>=0.43" "accelerate>=0.33" pyyaml
```

`torch` already ships with the Colab image; do not reinstall it.

## 2. Code and dataset

Mount Drive so the adapter survives the runtime being recycled:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload (or `git clone`) the repository, then install the package:

```python
%cd /content/osm-geoagent/finetuning
!pip install -q -e .
```

Copy one exported dataset version into `data/`. Export it beforehand on the
machine that has the Ariadne database:

```bash
./.venv/bin/python -m app.cli export-training-dataset --dataset-version v1
```

Five files must be present: three `.jsonl` splits, `.manifest.json` and
`.schema.json`.

## 3. Point the output at Drive

```python
import os
os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/ariadne-finetuning/artifacts"
os.environ["BASE_MODEL_ID"] = "Qwen/Qwen2.5-1.5B-Instruct"   # or whatever fits
```

## 4. Validate, baseline, train

```python
!python scripts/validate_dataset.py --config configs/qlora_analysis_plan.yaml
!python scripts/train_qlora.py --config configs/qlora_analysis_plan.yaml --baseline-only
!python scripts/train_qlora.py --config configs/qlora_analysis_plan.yaml
```

The first `train_qlora.py` call measures the untrained model on the frozen test
split. The second trains and re-measures on exactly the same examples, then
writes `report.json` and a registry entry.

## 5. Take the artifacts with you

The adapter is a few tens of megabytes, so downloading it directly is fine:

```python
!zip -r /content/adapter.zip "$OUTPUT_DIR/analysis-plan-qlora-001"
from google.colab import files
files.download('/content/adapter.zip')
```

## Notes

- Sessions are killed without warning. Writing to Drive is the only reliable
  persistence.
- If the model does not fit, lower `MAX_SEQ_LENGTH` first, then `LORA_R`.
  Raising `GRADIENT_ACCUMULATION_STEPS` keeps the effective batch size while
  reducing peak memory.
- Nothing here promotes a model. Bring the adapter and its report home, review
  them, and only then decide.
