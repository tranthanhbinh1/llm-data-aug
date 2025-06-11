# Dagster Integration Plan for LLM Data Augmentation Pipeline

## 1. Objectives
1. Replace ad-hoc Python scripts with a clear, asset-centric DAG (Directed Acyclic Graph) managed by [Dagster](https://dagster.io/).
2. Model the following domain assets and their lineage:
   * **`prompt_asset`** – The optimized prompt (text)
   * **`synthetic_data_asset`** – CSV containing LLM-generated reviews
   * **`score_asset`** – Numeric evaluation metric (e.g., weighted F1)
3. Enable incremental re-materialisation, monitoring, and selection of subsets of the graph using Dagster's asset selection syntax ([blog post](https://dagster.io/blog/updated-asset-selection-syntax)).

---

## 2. Current Code Snapshot
* Prompt generation lives in `src/prompt_optimization/*`.
* Synthetic data generation lives in `src/synthesizer/*`.
* Model training/evaluation lives in `src/trainers/*`.
* A Dagster skeleton already exists in `src/worker/`, but only contains stubs.

We will flesh out the skeleton with real `@asset` definitions and clean API boundaries while obeying project-wide rules (PEP8, ≤20 LOC per function, etc.).

---

## 3. Proposed Dagster Repository Layout
```
src/worker/
  __init__.py            # Defines Dagster repository
  resources/
    llm_resource.py      # Already present – wraps genai & instructor
  assets/
    prompt.py            # Defines prompt_asset
    synthetic_data.py    # Defines synthetic_data_asset
    score.py             # Defines score_asset
  jobs/
    full_pipeline.py     # Materialises the complete graph
  sensors/
    re_optimize.py       # Triggers when prompt_asset updates
  partitions/
    prompt_partitions.py # Dynamic partitions keyed by prompt hash
```

### 3.1 Asset Definitions
1. **Prompt Asset** (`prompt_asset`)
   * Inputs: `initial_prompt` (config), `improvement_request` (config)
   * Logic: Runs `PromptOptimizer.optimize`, returns best prompt as string.
   * Partitioning: dynamic by `(initial_prompt_hash, timestamp)` to keep history.
   * IO Manager: `StringIOManager` (built-in) or custom to store each prompt in `graphs/prompts/{partition_key}.txt`.

2. **Synthetic Data Asset** (`synthetic_data_asset`)
   * Inputs: `prompt_asset`, `sentiment` (partition or config)
   * Logic: Calls `AugGptRunner.generate_reviews_batch`, returns path to CSV.
   * IO Manager: `FileIOManager` writing to `data/llm_generated/{partition_key}.csv`.
   * Partitioning: match `prompt_asset` partition + sentiment.

3. **Score Asset** (`score_asset`)
   * Inputs: `synthetic_data_asset`
   * Logic: Chooses trainer (CNN-BERT Hybrid by default) and returns weighted F1 as float.
   * Materialisation policy: auto-materialise on upstream update.

### 3.2 Jobs & Graphs
* `full_pipeline_job` – a `define_asset_job` selecting all three assets; can be triggered via CLI, schedule, or sensor.
* Example selection: `dagster job materialize +prompt_asset` (materialise downstream assets).

### 3.3 Resources
* Re-use existing `LLMResource` for both prompt optimisation and synthetic generation.
* Add `TrainerResource` (encapsulates PyTorch device, tokenizer, hyper-params).

---

## 4. Implementation Steps

| Phase | Tasks |
|-------|-------|
| **0. Setup** | ➊ Add `dagster` and `dagster-webserver` to `pyproject.toml` (no `dagster-aws`). ➋ Ensure `.dagster_home` is initialised for local instance. |
| **1. Repository** | ➊ Implement `src/worker/__init__.py` with `@repository`. ➋ Register all assets & jobs. |
| **2. Assets** | ➊ Port logic into functions ≤20 LOC each. ➋ Greedy caching: if CSV already exists skip regeneration. ➌ Emit metadata (`MaterializeResult(metadata={"rows": len(df)})`). |
| **3. Resources** | ➊ Adapt `LLMResource` to return both instructor & raw client. ➋ Register in repository. |
| **4. Partitions & Policies** | ➊ Create dynamic partitions for prompts. ➋ Configure `AutoMaterializePolicy.eager()` for `score_asset`. |
| **5. Sensors & Schedules** | ➊ `re_optimize_sensor` – watches new data in `data/cleaned_user_reviews.csv` ⇒ rematerialise prompt. ➋ Daily schedule for `full_pipeline_job`. |
| **6. Local Testing** | ➊ `dagster dev` – visualise asset graph. ➋ Unit tests using `dagster._check.testing.build_assets_job` to run assets in-memory. |
| **7. CI/CD** | ➊ Add GitHub Action: `pytest`, `dagster-test`, and `dvc pull`/`dvc push` to manage data artifacts. ➋ Optionally containerise Dagster webserver via Docker Compose. |
| **8. Documentation** | ➊ Update `README.md` with Dagster usage. ➋ Provide asset selection cheatsheet (e.g., `key:"score_asset"+`). |

---

## 5. Design Considerations
1. **Artifact Storage & Versioning** – All assets are stored locally under `data/` or `graphs/` and tracked via **DVC**. Each asset function should `dvc add` new outputs (via `subprocess` call or a simple helper) and generate deterministic filenames (e.g., hash-prefix). Dagster asset metadata can include the corresponding DVC hash for lineage.
2. **Resource Re-use vs Isolation** – Long-running LLM sessions vs fresh per-asset calls; choose based on rate-limit behaviour.
3. **Determinism** – Seed random number generators inside assets to ensure reproducibility across re-materialisations.
4. **Caching** – Use Dagster's versioning: `@asset(code_version=...)` so unchanged prompt skips downstream work **and** rely on DVC's file hash to avoid duplicate storage.
5. **Metric Storage** – `score_asset` can `yield Output` with metadata (precision, recall) to enrich lineage views.
6. **Multi-objective Optimisation** – In future, add additional score assets (e.g., similarity) and combine via a **`composite_score_asset`**.
7. **Lineage Groups** – Use `group="generation"`, `group="evaluation"` for clarity.

---

## 6. Indicative Timeline
| Week | Deliverable |
|------|-------------|
| **1** | Dependencies installed; `dagster dev` renders empty repo. |
| **2** | Implement `prompt_asset`; unit test with static inputs. |
| **3** | Implement `synthetic_data_asset` & partition logic; data persists to disk. |
| **4** | Implement `score_asset`; full_pipeline_job runs end-to-end locally. |
| **5** | Add sensors, dynamic partitions, and auto-materialisation. |
| **6** | CI pipeline, documentation, and Dagster webserver deployment.

---

## 7. Risks & Mitigations
1. **LLM Rate Limits** – Mitigate via exponential back-off and Dagster retries (`RetryPolicy`).
2. **Large Local Artifacts** – Mitigate with DVC's deduplicated cache and `dvc gc` in CI; no external storage needed unless repository grows beyond capacity.
3. **Function Length Rule** – Break long logic into helper modules called by short asset functions.

---

## 8. References
* Dagster blog on asset selection syntax <https://dagster.io/blog/updated-asset-selection-syntax>
* Dagster guide on "Thinking in Assets" <https://dagster.io/blog/thinking-in-assets>
* Existing codebase modules: `src/prompt_optimization`, `src/synthesizer`, `src/trainers`

---

> This plan complies with workspace rules: modular design, PEP8, ≤20 LOC per function, max conditional depth 3.
