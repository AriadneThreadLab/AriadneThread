# Evaluation

Two evaluation sets, used for different questions.

| Set | Location | Comes from | Question it answers |
|---|---|---|---|
| Frozen test split | `data/<task>_<version>.test.jsonl` | Ariadne's exporter | Did fine-tuning improve on *this* dataset version? |
| Gold set | `evaluation/gold/analysis_plan_gold_v1.test.jsonl` | Hand-curated | Can the model plan the semantics we care about at all? |

Both are read-only inputs. Neither is ever fed back into training.

## Gold set

Six curated `user request -> AnalysisPlan` pairs covering the metric-selection
distinctions Ariadne actually depends on:

- "more parks" -> `count`
- "concentration" -> `density`
- "green-space provision" -> `total_area`
- "typical park size" -> `median_area`
- "average park size" -> `mean_area`
- "consistency" -> `standard_deviation` over a numeric property

Every assistant target was validated against the authoritative `AnalysisPlan`
model before it was committed, and `analysis_plan_gold_v1.schema.json` is that
model's generated JSON Schema.

## Reported metrics

`ariadne_finetuning.metrics` reports each dimension separately, never a single
blended accuracy:

| Metric | Meaning |
|---|---|
| `valid_analysis_plan_rate` | Generation parses *and* satisfies the plan schema |
| `analysis_type_accuracy` | `single_target` vs `comparison` |
| `target_preservation_accuracy` | All target labels kept, none invented or dropped |
| `radius_accuracy` | `radius_m` matches (comparison-plan task only) |
| `feature_concept_accuracy` | Feature concept matches after normalisation |
| `comparison_goal_accuracy` | Stated goal overlaps the reference goal |
| `metric_intent_accuracy` | Primary metric matches |
| `hallucinated_field_rate` | Share of generations with fields the schema forbids |
| `invented_coordinate_rate` | Share of generations with coordinates absent from the reference |

A metric that never applied to a dataset reports `null` with an
`applicability` count of `0`, so "not measured" is never mistaken for "scored
zero".

## Frozen test set

`DatasetBundle.test_digest` fingerprints the test split. Baseline and
post-training reports both carry it, and `evaluation.compare` refuses to
produce a before/after table when the two digests differ.
