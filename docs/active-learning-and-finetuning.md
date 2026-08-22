# Active learning and fine-tuning

How Ariadne turns real interactions into curated training data, and how a model
is trained from it — in two systems that never call each other.

## Five words that get confused

| Term | What it actually is | Where it lives |
|---|---|---|
| **RAG** | Runtime knowledge retrieval. The model reads OSM documentation while answering. | `app/rag/`, `search_osm_knowledge` |
| **Active learning** | Selecting the most informative interactions for a human to review. | `app/active_learning/` |
| **Feedback loop** | Collecting real reactions to real answers. | `POST /api/v1/agent/feedback`, CLI review |
| **Fine-tuning** | Changing model behaviour using curated examples. | `finetuning/` |
| **Evaluation** | Measuring whether the new model is actually better. | `finetuning/evaluation/`, `ariadne_finetuning.metrics` |

RAG changes what the model *knows this second*. Fine-tuning changes what the
model *is*. Active learning decides which interactions are worth a human's
attention in between.

## The flow

```
User
 ↓
Ariadne agent (bounded planner-executor loop)
 ↓
Operational trace + validated plan + tool outcomes
 ↓
Active learning scorer          ← deterministic, no LLM involved
 ↓                                (below threshold: discarded)
Candidate stored (review_status = pending)
 ↓
Human review queue: approve | correct | reject
 ↓
Curated dataset export → data/analysis_plan_vN.{train,validation,test}.jsonl
 ↓                        + manifest.json (lineage) + schema.json (authoritative)
QLoRA supervised fine-tuning (separate pipeline, separate dependencies)
 ↓
Evaluation on the frozen test split: baseline vs adapter
 ↓
Manual promotion decision
```

The production service never trains anything. It selects, stores and exports.

## Part 1 — Active learning inside Ariadne

### Where it hooks in

The orchestrator is unchanged in substance: it records operational trace
events, as it always did. The API layer offers each finished run to
`ActiveLearningService.observe_run`, which decides on its own whether the run is
worth retaining. Selection failures are logged and swallowed, so a bug in this
subsystem can never turn a good answer into an HTTP error.

### Package layout

| Module | Responsibility |
|---|---|
| `contracts.py` | Candidate model, reason/status enums, repository protocol |
| `sanitization.py` | Reject hidden reasoning, redact credentials |
| `signals.py` | Trigger extraction from the trace; model vs external attribution |
| `scoring.py` | Deterministic informativeness score |
| `novelty.py` | Query hash, task signature, near-duplicate detection |
| `service.py` | Selection, feedback, review lifecycle |
| `repository.py` | In-memory and PostgreSQL stores |
| `export.py` | Grouped splitting, JSONL export, manifest, lineage |
| `cli_commands.py` | Review-queue and export commands |

### What a candidate holds

Request metadata (`request_id`, `created_at`, `model_id`, `prompt_version`,
`tool_schema_version`), the user query, task type and task signature, the
validated `analysis_plan` and `comparison_plan`, the final answer, the tool
sequence, tool validation events, place-resolution events, a compact RAG
evidence summary, the metric selection summary, error codes split into
model-attributed and external, warnings, selection reasons, the informativeness
score, review state, any human correction, dataset split and schema version.

It does **not** hold GeoJSON, prompts, retrieved passage bodies, credentials or
model reasoning.

### Selection reasons

Bounded enum, never free text:

`TOOL_VALIDATION_FAILED`, `INVALID_STRUCTURED_OUTPUT`, `WRONG_OR_INELIGIBLE_TOOL`,
`PLACE_RESOLUTION_AMBIGUOUS`, `PLACE_RESOLUTION_FAILED`, `GROUNDING_CONFLICT`,
`METRIC_SELECTION_UNCERTAIN`, `METRIC_FEASIBILITY_FAILED`, `GUARDRAIL_TRIGGERED`,
`HALLUCINATED_COORDINATE_BLOCKED`, `HALLUCINATED_TAG_BLOCKED`,
`DEPENDENCY_ORDER_VIOLATION`, `LLM_PROTOCOL_ERROR`, `MODEL_TIMEOUT`,
`USER_NEGATIVE_FEEDBACK`, `HUMAN_CORRECTION_AVAILABLE`, `MODEL_DISAGREEMENT`,
`NOVEL_QUERY`, `SUCCESSFUL_HIGH_VALUE_TRACE`.

### Model errors versus external errors

This distinction is the heart of the scorer. An Overpass 504 says nothing about
the model; a rejected `place_ref` says a great deal.

| Observation | Attribution | Effect |
|---|---|---|
| `overpass_timeout`, `overpass_rate_limited`, `overpass_upstream_error` | external | Score capped at 0.05, not retained |
| `ollama_unreachable`, `dependency_unavailable` | external | Same |
| Nominatim transport failure | external | Same |
| Nominatim answered but nothing matched the model's query | model | `PLACE_RESOLUTION_FAILED` |
| `tool_argument_error` | model | `TOOL_VALIDATION_FAILED` (+ guardrail reasons) |
| Invented coordinates blocked | model | `HALLUCINATED_COORDINATE_BLOCKED` |
| Grounded tag substituted | model | `HALLUCINATED_TAG_BLOCKED`, `GROUNDING_CONFLICT` |
| Unknown `place_ref` / `dataset_ref` | model | `DEPENDENCY_ORDER_VIOLATION` |
| Backend fell back to a deterministic plan | model | `INVALID_STRUCTURED_OUTPUT`, `MODEL_DISAGREEMENT` |

### Scoring

Deterministic and explainable. Weights are summed, capped at 1.0, then scaled
by a novelty factor:

```
score = min(1, Σ weight(reason)) × novelty_factor
novelty_factor = 1.0                                  when nothing similar is stored
               = max(0.35, 1 − 0.25 × duplicate_count) otherwise
```

Highest weights go to a human correction (0.60), explicit negative feedback
(0.50) and a blocked invented coordinate (0.45). A local model timeout is worth
only 0.15: it is mostly a hardware fact. An ordinary success scores 0.25 and
therefore needs novelty (+0.20) to clear the 0.20 storage threshold — which is
exactly why the second identical success is dropped.

The scorer never asks an LLM whether a sample is interesting, and never reads a
model's self-reported confidence, because local 7B confidence is not calibrated.

### Diversity

Three cheap layers run on the request path: a normalized query hash, a
structural task signature such as `comparison|park|radius|count`, and token-set
overlap for near-duplicate wording. Differing numbers (a 2 km versus a 5 km
comparison) always break a match, because they change the expected plan.

A fourth, optional layer reuses the existing BGE-M3 infrastructure through the
`SemanticSimilarityIndex` protocol. It is deliberately not wired into request
handling: embedding a query costs more than everything else here, so semantic
enrichment belongs in a batch job.

### Thresholds

| Setting | Default | Meaning |
|---|---|---|
| `ACTIVE_LEARNING_ENABLED` | `true` | Master switch |
| `ACTIVE_LEARNING_MIN_SCORE_TO_STORE` | `0.20` | Below this, the run is discarded |
| `ACTIVE_LEARNING_MIN_SCORE_FOR_REVIEW` | `0.35` | Review-queue cutoff |
| `ACTIVE_LEARNING_NOVELTY_THRESHOLD` | `0.85` | Token overlap counting as a duplicate |
| `ACTIVE_LEARNING_MAX_DUPLICATES_PER_SIGNATURE` | `5` | Cap per task signature |
| `ACTIVE_LEARNING_RECENT_RUN_CACHE` | `256` | Recent runs kept so late feedback can still materialise a candidate |

### Successes matter too

A dataset of nothing but failures teaches a model to fail politely. Clean runs —
correct multi-target comparison, preserved target names, sensible metric,
correct tool ordering, no invented coordinates — are captured as
`SUCCESSFUL_HIGH_VALUE_TRACE` and become the positive examples. Failures are
useful only *after* a human corrects or explicitly approves them.

### Review lifecycle

```
pending ──approve──▶ approved ──export──▶ exported
   │  └──correct──▶ corrected ──export──▶ exported
   └──reject───▶ rejected
```

`approved_for_training` requires an approved or corrected review *and* either a
validated successful output or a human-corrected target. Approving a failed run
as-is raises `ReviewTransitionError`; the model itself re-validates the rule, so
it cannot be bypassed by constructing a candidate directly. An exported
candidate is frozen: later feedback annotates it but does not reopen it.

### Feedback

```bash
curl -X POST localhost:8100/api/v1/agent/feedback -H 'content-type: application/json' -d '{
  "request_id": "…", "sentiment": "negative",
  "failure_category": "wrong_metric",
  "note": "density would have been the right indicator"
}'
```

Feedback queues a run for review. It never approves anything and never triggers
training. If the run was filtered out earlier, the bounded recent-run cache lets
the service materialise the candidate retroactively.

### CLI

```bash
python -m app.cli list-active-learning-candidates --review-queue
python -m app.cli show-active-learning-candidate --candidate-id <id>
python -m app.cli approve-active-learning-candidate --candidate-id <id> --reviewer <name>
python -m app.cli correct-active-learning-candidate --candidate-id <id> \
    --corrected-output-file plan.json --reviewer <name>
python -m app.cli reject-active-learning-candidate --candidate-id <id> --note "duplicate"
python -m app.cli active-learning-stats
python -m app.cli export-training-dataset --dataset-version v1 --dry-run
```

A corrected plan is validated against the authoritative model at correction
time, not silently at export time.

### Persistence

Three tables, separate from agent execution state
(`alembic/versions/20260816_0002_active_learning_tables.py`):

| Table | Holds |
|---|---|
| `active_learning_candidates` | One row per retained run |
| `active_learning_reviews` | Append-only audit of review decisions |
| `training_dataset_exports` | Dataset lineage: version, task, candidate ids, request ids, digest |

### Privacy

Two different guarantees, deliberately:

- **Hidden reasoning is rejected, never cleaned.** A candidate that would carry
  `<think>` or `reasoning_content` is refused outright, so the failure is
  visible rather than half-applied.
- **Credentials are redacted in place.** API keys, bearer tokens, `hf_`/`ghp_`
  tokens, credentialed URLs and `key=value` secrets become `[redacted:…]`,
  because a user request may legitimately contain a token-shaped string.

Both run at model-validation time, which means they apply on write *and* on
read from the database.

## Part 2 — Dataset export

### Target task

`user request → AnalysisPlan`, using the exact `AnalysisPlan` model that
`analyze_features` validates. No parallel training schema exists anywhere.
`MultiTargetComparisonPlan` is available as a second task for the
radius-bearing planner schema.

Because `AnalysisPlan` binds targets to backend-generated dataset references,
the exported prompt states which references exist — rendered from the validated
plan itself:

```
Compare public parks within 2 km of University of Tehran and Sharif University of Technology.

Available datasets:
- osm_result_1: University of Tehran
- osm_result_2: Sharif University of Technology

Return one JSON object matching the Ariadne AnalysisPlan schema. Use only the dataset references listed above.
```

The assistant target is the canonical JSON of the validated plan, and nothing
else. Export metadata lives in the manifest, never inside the target.

### Splitting

Candidates are clustered before they are split: two candidates join the same
group when they share a task signature or when their queries are
near-duplicates. Whole groups are then assigned by hashing
`dataset_version:group_key`, so a near-duplicate can never straddle train and
test, and the same dataset version always produces the same test set. Export is
byte-for-byte deterministic for a fixed version and candidate set.

### Lineage

The manifest records dataset version, task, target schema name, export
timestamp, per-split candidate ids, source models, prompt versions, reviewer
statuses, skipped candidates with reasons, and a content digest over all three
splits.

## Part 3 — The fine-tuning pipeline

See [finetuning/README.md](../finetuning/README.md) and
[finetuning/COLAB.md](../finetuning/COLAB.md).

The pipeline is a separate top-level directory with its own dependencies. It
reads exported files and never touches the Ariadne database. Tests enforce
this: `app/` may not import a training package, `ariadne_finetuning` may not
import `app`, SQLAlchemy or FastAPI, and no ML package may be imported at
module scope.

Baseline evaluation runs *before* training, on the frozen test split, so the
post-training number means something. `evaluation.compare` refuses to build a
before/after table unless both reports carry the same test digest and the same
prompts in the same order.

Every metric is reported separately, with an applicability count, so a metric
that never applied is never mistaken for one that scored zero.

## Part 4 — Promotion

`train → evaluate → manual approval → optional merge/export → optional serving
→ A/B evaluation`.

Every experiment is recorded as `experimental` in
`finetuning/artifacts/registry.json`. Promotion is a human decision recorded by
`scripts/promote.py`, and even an `approved` experiment changes nothing until
someone deliberately points Ariadne at a new model.

## What this design refuses to do

- Train during a request, on a schedule, or in a background task
- Approve a sample because the model said it was confident
- Export an unreviewed failed output
- Treat an Overpass outage as evidence about the model
- Store chain-of-thought anywhere, for any reason
- Promote a model because a metric improved
