# Reasoning Trace Pipeline

Research pipeline for detecting and mitigating pathological reasoning loops in long LLM chains-of-thought.

Core objective: identify and reduce dead-compute patterns such as repetitive reflection, excessive re-verification, and latent-state orbits/fixed points.

---

## Project Scope

This project implements a **three-layer diagnosis stack** plus **active mitigation**:

1. **Semantic Layer (Cognitive Flow)**
   - Labels each reasoning step with functional role:
     - `ProblemSetup`, `Calculation`, `Verification`, `Interpretation`, `Conclusion`, `Other`
   - Builds a state-transition matrix and computes self-transition rate to flag loops.

2. **Structural Layer (DAG Diagnosis)**
   - Converts segmented trace into a DAG.
   - Labels nodes as `Progress` vs `Review`.
   - Supports graph pruning of low-value review branches.

3. **Latent Layer (Mechanistic Monitoring)**
   - Uses hidden-state trajectory hooks to detect convergence behaviors:
     - `FIXEDPOINT`, `ORBIT`, `SLIDER`, `UNKNOWN`.
   - Uses cosine-similarity recurrence analysis and optional FFT peak detection.

4. **Mitigation Layer**
   - Branch-level and depth-level DAG pruning.
   - CCD-style decoding-time intervention for low-confidence steps.
   - First-step quality filter (PRM-style judge scoring).
   - Self-braking epiphany sentence injection when loops are detected.

---

## Repository Layout

- `models.py`
  - Local model wrapper (`torch` + `mlx`) with optional hidden-state hooks.
  - `<think>...</think>` extraction helpers.
- `segmentation.py`
  - Segmenter interface and implementations:
    - `DelimiterSegmenter`
    - `KeywordSegmenter`
    - `HybridSegmenter`
    - `LLMGraphSegmenter`
  - `build_segmenter(...)` factory.
- `diagnostics.py`
  - `CognitiveFlowAnalyzer` (judge-based labels + transition matrix).
  - `GraphConstructor` (DAG with `Progress`/`Review` roles).
- `mechanistic.py`
  - `LatentMonitor` for fixedpoint/orbit/slider classification.
- `mitigation.py`
  - `PruningEngine`, `ConfusionContrastiveLogitsProcessor`, `FirstStepFilter`.
- `pipeline.py`
  - `ReasoningPipeline` orchestration and `AtomicReasoningUnit`.
  - Combines diagnosis + mitigation and emits rich metadata.
- `main.py`
  - Single-run CLI for prompt-level debugging and trace inspection.
- `prepare_prompts.py`
  - Extracts prompts from `self_correction_llms` data using source prompt templates.
- `experiments.py`
  - Batch experiments across prompts, segmenters, and run modes (`baseline`, `mitigated`).
  - Writes `per_run.jsonl` and aggregated `summary.json`.
- `report.py`
  - Post-processing summary report for baseline-vs-mitigated deltas and ranking.

---

## Hardware + Runtime Notes

- Target hardware: Apple Silicon (MacBook Air M4), MPS/MLX.
- `torch` backend supports:
  - MPS inference
  - optional 4-bit quantization (environment-dependent)
  - hidden-state trajectory extraction for latent analysis
- `mlx` backend supports efficient Apple local inference (hidden-state hooks may be limited).

---

## End-to-End Setup (Conda)

### 1) Create environment

```bash
conda create -n reasoning-trace python=3.11 -y
conda activate reasoning-trace
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install torch transformers accelerate numpy
```

Optional:

```bash
pip install mlx-lm
pip install bitsandbytes
```

### 3) Verify runtime

```bash
python -c "import torch; print('mps_available=', torch.backends.mps.is_available())"
```

---

## End-to-End Workflow (What To Do)

### Step A: Prepare prompt set

Generate `prompts.json` directly from `self_correction_llms`:

```bash
python prepare_prompts.py \
  --source-root /Users/blizzard/Documents/Projects/self_correction_llms \
  --data-names gsm8k,math \
  --split test \
  --prompt-type deepseek-r1 \
  --num-test-sample 200 \
  --output-json prompts.json \
  --output-records-json prompts_records.json
```

If a dataset name is wrong for your local source data, the script now prints available datasets for the selected split. You can also skip missing datasets:

```bash
python prepare_prompts.py \
  --source-root /Users/blizzard/Documents/Projects/self_correction_llms \
  --data-names gsm8k,math500 \
  --split test \
  --prompt-type deepseek-r1 \
  --on-missing skip \
  --output-json prompts.json
```

Or create `prompts.json` manually:

```json
[
  "Solve this arithmetic problem ... use <think> tags.",
  "Solve this algebra problem ... use <think> tags.",
  "Solve this probability problem ... use <think> tags."
]
```

or `prompts.txt` (one prompt per line).

### Step B: Sanity-check single run

```bash
python main.py \
  --segmenter graph \
  --backend torch \
  --judge-backend torch \
  --enable-diagnostics \
  --enable-latent \
  --enable-pruning \
  --enable-first-step-filter \
  --enable-ccd \
  --output-json outputs/run_debug.json
```

Check `outputs/run_debug.json` for:
- segmented `units`
- `metadata.loop_analysis`
- `metadata.cognitive_flow`
- `metadata.dag`
- `metadata.latent`
- `metadata.first_step_filter`

### Step C: Run full baseline-vs-mitigated batch

```bash
python experiments.py \
  --prompts-file prompts.json \
  --segmenters delimiter,keyword,hybrid,graph \
  --run-modes baseline,mitigated \
  --backend torch \
  --judge-backend torch \
  --enable-diagnostics \
  --enable-latent \
  --enable-pruning \
  --enable-first-step-filter \
  --enable-ccd \
  --output-dir outputs/exp_compare
```

Outputs:
- `outputs/exp_compare/per_run.jsonl`
- `outputs/exp_compare/summary.json`

### Step D: Generate ranked delta report

```bash
python report.py \
  --summary-json outputs/exp_compare/summary.json \
  --output-path outputs/exp_compare/report.md
```

Read:
- `outputs/exp_compare/report.md`

### Step E: Iterate

1. Start with `graph` and `hybrid` segmenters.
2. Compare `baseline:graph` vs `mitigated:graph`.
3. Tune thresholds/prompts/judge model.
4. Scale prompt set and rerun experiments.

---

## CLI Reference

### `main.py` (single-run)

Key flags:
- `--prompt`
- `--segmenter` (`delimiter|keyword|hybrid|graph`)
- `--backend` (`torch|mlx`)
- `--judge-backend` (`torch|mlx`)
- `--primary-model`
- `--judge-model`
- `--max-new-tokens`
- `--return-hidden-states`
- `--enable-diagnostics`
- `--enable-latent`
- `--enable-pruning`
- `--enable-first-step-filter`
- `--enable-ccd`
- `--output-json`

### `experiments.py` (batch compare)

Key flags:
- `--prompts-file` or repeated `--prompt`
- `--segmenters`
- `--run-modes` (use `baseline,mitigated`)
- `--enable-diagnostics`
- `--enable-latent`
- `--enable-pruning`
- `--enable-first-step-filter`
- `--enable-ccd`
- `--output-dir`

### `prepare_prompts.py` (source-aligned prompt extraction)

Key flags:
- `--source-root` (path to `self_correction_llms`)
- `--data-dir` (optional override, default `<source-root>/data`)
- `--data-names` (comma-separated dataset names)
- `--split`
- `--prompt-type` (must exist in source `PROMPT_TEMPLATES`)
- `--num-test-sample`, `--start`, `--end`
- `--output-json`
- `--output-records-json`

### `report.py` (post-analysis)

Key flags:
- `--summary-json`
- `--output-path`

---

## Metrics and How To Interpret Them

Primary loop metrics:
- `avg_repeated_unit_ratio` (lower better)
- `avg_adjacent_overlap_ratio` (lower better)
- `avg_cognitive_self_transition_rate` (lower better)
- `loop_detected_rate` (lower better)

Structural metrics:
- `avg_prune_ratio` (higher means more review-heavy branches removed)
- `judge_insert_rate` vs `judge_merge_rate` (healthy progress should not collapse into merge-only traces)

Latent metrics:
- `latent_classification_counts`:
  - want fewer `FIXEDPOINT` / `ORBIT`
  - `SLIDER`/`UNKNOWN` trends can still be inspected manually

Quality guardrail:
- `avg_first_step_score` should not fall significantly while loop metrics improve.

---

## Suggested Ablation Plan

1. **Segmentation only**: `run-modes baseline`, compare segmenters.
2. **Add diagnostics**: enable cognitive + DAG metrics (no pruning).
3. **Add latent monitoring**: check fixedpoint/orbit prevalence.
4. **Add mitigations incrementally**:
   - pruning only
   - pruning + first-step filter
   - pruning + first-step filter + CCD
5. Compare all against baseline with `report.py`.

---

## Troubleshooting

- If MPS is unavailable:
  - verify PyTorch install and run on CPU as fallback.
- If bitsandbytes fails on macOS:
  - keep running without 4-bit quantization or switch to MLX path.
- If `graph` segmentation fails:
  - ensure judge model can be loaded (`--judge-model`, `--judge-backend`).
- If latent classification is mostly `UNKNOWN`:
  - increase generation length (`--max-new-tokens`) to get longer hidden-state trajectories.

---

## Minimal Copy/Paste Commands

Setup:

```bash
conda create -n reasoning-trace python=3.11 -y
conda activate reasoning-trace
python -m pip install --upgrade pip
pip install torch transformers accelerate numpy
```

Batch run:

```bash
python experiments.py \
  --prompts-file prompts.json \
  --segmenters delimiter,hybrid,graph \
  --run-modes baseline,mitigated \
  --backend torch \
  --judge-backend torch \
  --enable-diagnostics \
  --enable-latent \
  --enable-pruning \
  --enable-first-step-filter \
  --enable-ccd \
  --output-dir outputs/exp_compare
```

Extract prompts from source project:

```bash
python prepare_prompts.py \
  --source-root /Users/blizzard/Documents/Projects/self_correction_llms \
  --data-names gsm8k \
  --split test \
  --prompt-type deepseek-r1 \
  --num-test-sample 200 \
  --output-json prompts.json
```

Report:

```bash
python report.py \
  --summary-json outputs/exp_compare/summary.json \
  --output-path outputs/exp_compare/report.md
```
