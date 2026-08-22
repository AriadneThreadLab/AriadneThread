# Exported datasets

Ariadne writes versioned dataset artifacts here. The files themselves are
gitignored: they are reproducible outputs of approved review decisions, not
repository content.

```bash
# from the repository root, using the application virtualenv
./.venv/bin/python -m app.cli export-training-dataset --dataset-version v1
```

Each version produces five files:

| File | Contents |
|---|---|
| `analysis_plan_v1.train.jsonl` | Conversational SFT samples |
| `analysis_plan_v1.validation.jsonl` | Conversational SFT samples |
| `analysis_plan_v1.test.jsonl` | Frozen evaluation samples |
| `analysis_plan_v1.manifest.json` | Lineage: candidate ids, request ids, splits, reviewer statuses, source models, digest |
| `analysis_plan_v1.schema.json` | Generated JSON Schema of the authoritative target model |

Sample shape (metadata never appears inside the assistant target):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"analysis_type\":\"comparison\",...}"}]}
```

Validate a directory before training:

```bash
python scripts/validate_dataset.py --config configs/qlora_analysis_plan.yaml
```
