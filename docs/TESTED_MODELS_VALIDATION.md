# Adaptive Testing & Automated Safety Validation for LLM Inference Configuration

## Executive Summary

This document describes a novel system for validating large language model (LLM) inference configurations against empirically tested operating boundaries. The core challenge: LLM inference performance is a complex, non-linear function of dozens of configuration parameters (tensor parallelism, pipeline parallelism, sequence lengths, batch sizes, quantization, backend versions, hardware SKUs, etc.), with performance "knees"—sharp dropoffs—occurring at different points in the parameter space depending on complex interactions between parameters.

**The Problem**: Existing approaches either reject all untested configurations (too conservative) or accept configurations outside tested regions without validation (too permissive). Users lack actionable feedback when their configuration risks poor performance.

**The Solution**: A three-phase adaptive testing regime that systematically samples the parameter space to identify performance knees, combined with an automated knee-detection algorithm and a machine-learning classifier (starting with Decision Trees, upgrading to XGBoost/CatBoost only if needed) that predicts whether a user's configuration is **within tested performance regions**.

---

## Problem Statement

### What We're Solving

LLM inference is sensitive to configuration. For a given model, hardware, and backend combination, users select values for parameters like:
- Parallelism strategy (`tp`, `pp`, `ep`, etc.)
- Sequence length constraints (`isl`, `osl`)
- Quantization precision
- Concurrency / batch sizing
- Paging and memory strategies
- Backend version and compilation flags

Each parameter affects throughput, latency, and memory footprint. **But the effects are deeply non-linear and interactive.**

For example:
- At `tp=2, concurrency=8`, a system may deliver 400 tok/s
- Increasing to `tp=4, concurrency=8` might improve to 550 tok/s (scaling well)
- But `tp=4, concurrency=16` might drop to 300 tok/s (communication overhead, memory pressure)
- And `tp=8, pp=4, isl=32k, concurrency=32` may cause OOM or performance collapse

**Current state**: ConfigIQ flags configurations as "tested" or "untested" based on a binary list. This is insufficiently granular. A configuration could be "untested" but still within performance boundaries, or theoretically valid but outside empirically tested regions.

**User impact**: 
- Users make recommendations based on incomplete information
- Support burden: "Why is my config flagged?" with no clear answer
- Business risk: recommendations that look right but perform poorly in production

### Why This Is Hard

1. **Parameter space is high-dimensional** (8–15 continuous/categorical parameters per model/hw/backend)
2. **Interactions are complex** (performance depends on products, ratios, and non-monotonic relationships)
3. **Performance knees are model/hardware/backend-specific** (what causes a cliff on V100s may not on H100s)
4. **Testing is expensive** (each config requires 5–15 min of GPU time to benchmark reliably)
5. **Extrapolation fails** (interpolating between tested points works; extrapolating does not)

Conventional ML approaches struggle because:
- Simple rules ("if `tp > 64`, reject") miss parameter interactions
- Surrogate models trained on sparse data extrapolate poorly
- Hand-coded constraints are brittle and don't adapt as backends evolve

---

## Proposed Solution: Three-Phase Adaptive Testing

We propose a systematic testing regime that adaptively samples the parameter space, identifies performance knees via automated algorithms, and trains a classifier to predict whether arbitrary user configurations are within tested performance regions.

### Phase 1: Fractional Factorial Coverage (90–150 runs)

**Goal**: Obtain broad coverage of the parameter space and identify regions of interest (performance cliffs, OOM boundaries).

**Method**: Fractional factorial design (FFD) with statistical confounding to reduce the search space:

1. Select key parameters for a single (model, hardware, backend) triple and their ranges
2. Generate a fractional factorial design matrix that tests ~1/2 to 1/3 of all possible combinations
3. Run guidellm benchmarks for each configuration
4. Record throughput, latency (p50, p99), memory utilization, efficiency, and success/failure status

**Why FFD?**
- Captures main effects and first-order interactions with ~70% fewer runs
- Mathematically principled: confounding structure is designed to preserve validity
- Standard in industrial engineering; proven for complex systems
- Baseline for adaptive refinement

**Output**: 90–150 runs per (model, hardware, backend) providing the initial parameter sensitivity landscape.

---

### Phase 2: Adaptive Knee-Seeking (50–100 runs)

**Goal**: Densify sampling around performance transitions and identify exact OOM/performance boundaries.

**Method**: Analyze Phase 1 results and adaptively sample:

1. **Identify high-curvature regions** using smoothness metrics:
   - Compute local throughput gradients along each parameter dimension
   - Flag regions where ∇throughput exceeds threshold (e.g., >30% drop over parameter interval)

2. **Binary-search for boundaries**:
   - For parameters where Phase 1 shows steep dropoff, binary-search for the exact threshold
   - Example: if `concurrency=8 → 16` drops 40%, test {12, 14, 15} to find the knee

3. **Interaction hotspots**:
   - Identify parameter pairs with high interaction (e.g., `tp × concurrency`)
   - Add runs that test these pairs at different levels

4. **Edge cases**:
   - Test absolute min/max for each parameter
   - Test combinations suspected to cause OOM based on memory models

**Output**: 50–100 additional runs providing dense coverage of critical transition regions.

---

### Phase 3: Boundary Validation (20–50 runs)

**Goal**: Confirm hard limits and validate edge behavior; ensure classifier training has high confidence at boundaries.

**Method**:
1. Test at boundaries identified in Phase 2
2. Confirm reproducibility (repeat a few challenging configs to check variance)
3. Identify "performance margin" regions (e.g., configs >10% below OOM threshold)

**Output**: 20–50 runs confirming boundary behavior.

---

## Automated Knee-Detection Algorithm

Once Phases 1–3 are complete, we automatically label each run as `within_tested_region: true/false` using a multi-criteria algorithm that detects performance knees and out-of-bounds conditions.

### Algorithm: Knee Detection via Performance Curvature & Efficiency Thresholds

**Input**: Sorted dataset of (configuration, throughput, memory_util, latency_p99, success) tuples for a single (model, hardware, backend).

**Output**: Binary label `within_tested_region` for each row.

#### Step 1: Filter Invalid Configurations
```
For each row:
  if success == false OR memory_util > 95%:
    within_tested_region = false
    goto Step 4
```

Rationale: OOM, crashes, and near-OOM states are outside tested operating regions by definition.

---

#### Step 2: Compute Local Performance Smoothness (Knee Detection)

For each configuration c in the dataset, compute how "smooth" throughput is in the neighborhood of c. High curvature indicates proximity to a performance knee. This threshold is automatically calibrated per dataset—no fixed tuning.

```
For each parameter p:
  configs_lower = [x in dataset | x differs from c only in p, x[p] < c[p], 
                   maximize x[p]]
  configs_upper = [x in dataset | x differs from c only in p, x[p] > c[p],
                   minimize x[p]]
  
  if configs_lower and configs_upper exist:
    throughput_lower = max(configs_lower).throughput
    throughput_center = c.throughput
    throughput_upper = min(configs_upper).throughput
    
    delta_p_lower = c[p] - max(configs_lower)[p]
    delta_p_upper = min(configs_upper)[p] - c[p]
    
    # Second derivative approximation (curvature)
    curvature_p = |second_derivative(throughput, p)| 
                ≈ |(throughput_upper - throughput_center) / delta_p_upper 
                   - (throughput_center - throughput_lower) / delta_p_lower|
    
    curvature_scores[p] = curvature_p

# Adaptive threshold: flag if in top quartile of curvature for this dataset
max_curvature = max(curvature_scores.values())
curvature_75th_percentile = percentile(all_curvature_scores, 75)

if max_curvature > curvature_75th_percentile:
  within_tested_region = false
  goto Step 4
```

**Why this is adaptive**: Instead of a fixed threshold (e.g., "0.15"), we use the dataset's own curvature distribution. A hardware platform with steep throughput curves has higher absolute curvatures but uses the same percentile threshold. The algorithm self-calibrates per model/hardware/backend combo.

---

#### Step 3: Efficiency Comparison

Compare the configuration's efficiency (throughput per unit cost: tokens/s per allocated GPU resource) to the cohort's efficiency distribution. This threshold adapts automatically to the dataset—no hand-tuning required.

```
efficiency[c] = throughput[c] / (num_gpus × utilization_factor)

cohort_median_efficiency = median([efficiency[x] for x in dataset])
cohort_std = stddev([efficiency[x] for x in dataset])

efficiency_z_score = (efficiency[c] - cohort_median_efficiency) / cohort_std

if efficiency_z_score < -1.5:  # More than 1.5 std below cohort median
  within_tested_region = false
  goto Step 4
```

**Why this is adaptive**: The 1.5 std threshold is relative to the variance *in this dataset*. A hardware platform with high throughput variance has a wider acceptance band; one with tight variance has a narrower band. The threshold self-calibrates per model/hardware/backend combo.

---

#### Step 4: Latency Stability Check

Ensure p99 latency is not pathologically high (indicating queueing, resource contention, or GC overhead). The baseline automatically adapts to the dataset.

```
# Baseline: median p99 latency among high-throughput, stable configs
latency_p99_baseline = median([latency_p99[x] for x in dataset 
                                if throughput[x] > 0.9 * max_throughput])

# Adaptive threshold: flag if significantly higher than stable baselines
if c.latency_p99 > 3.0 * latency_p99_baseline:
  within_tested_region = false
```

**Why this is adaptive**: The 3.0× multiplier is relative to the baseline latency in *this dataset*. Hardware with inherently higher latency has a higher absolute threshold but the same relative margin. The algorithm scales across different hardware without tuning.

---

#### Step 5: Conservative Performance Margin

Apply a conservative margin to account for variance and extrapolation risk:

```
efficiency_margin = 0.85  # Require at least 85% of median cohort efficiency

if efficiency[c] / cohort_median_efficiency < efficiency_margin:
  within_tested_region = false
```

---

#### Step 6: Consensus Label

```
if any of Steps 1, 3, 4, 5 flagged the config as outside tested region:
  within_tested_region = false
else:
  within_tested_region = true
```

---

### Algorithm Properties

- **Fully Automated**: No manual tuning or per-combo calibration required. All thresholds are relative metrics computed from the dataset itself.
- **Interpretable**: Each flag has a clear reason (OOM, efficiency drop, curvature spike, latency spike)
- **Adaptive**: Thresholds are cohort-relative (percentiles, z-scores, multiples of baseline), not fixed values. Algorithm self-calibrates for each model/hardware/backend.
- **Robust**: Uses multiple independent signals (throughput curvature, efficiency distribution, latency stability) to reduce false positives; consensus labeling prevents noise from triggering flags
- **Conservative**: Errs toward marking as outside tested region when uncertain; includes safety margin (Step 5)

---

## Dataset Schema

For each (model, hardware, backend) triple, we collect a dataset of the form:

```json
{
  "id": "llama70b-8xh100-vllm062-run0042",
  "metadata": {
    "model": {
      "name": "meta-llama/Llama-2-70b-hf",
      "context_length": 4096,
      "n_layers": 80,
      "hidden_dim": 8192,
      "n_heads": 64,
      "n_kv_heads": 8
    },
    "hardware": {
      "gpu_type": "H100",
      "gpu_count": 8,
      "gpu_memory_gb": 80,
      "interconnect": "NVLink",
      "node_memory_gb": 960
    },
    "backend": {
      "name": "vLLM",
      "version": "0.6.2",
      "build_flags": ["flash_attn=true", "cudnn_benchmark=true"],
      "dtype": "float16"
    },
    "timestamp": "2026-08-24T10:30:00Z",
    "benchmark_tool": "guidellm",
    "benchmark_duration_sec": 300
  },
  
  "configuration": {
    "parallelism": {
      "tensor_parallel_size": 4,
      "pipeline_parallel_size": 2,
      "expert_parallel_size": 1,
      "distributed_executor_backend": "ray"
    },
    "sequence_lengths": {
      "input_seq_len_min": 512,
      "input_seq_len_max": 8192,
      "output_seq_len": 128,
      "sliding_window": null
    },
    "quantization": {
      "weight_quantization": "int8",
      "kv_quantization": "int8",
      "activation_quantization": null
    },
    "concurrency": {
      "max_num_seqs": 32,
      "max_model_len": 8192,
      "enable_chunked_prefill": true,
      "prefill_chunk_size": 4096
    },
    "cache_management": {
      "block_size": 16,
      "gpu_memory_utilization": 0.9,
      "cpu_offload_gb": 0,
      "enable_prefix_caching": false
    },
    "scheduling": {
      "enable_speculative_decoding": false,
      "num_speculative_tokens": 0,
      "num_lookahead_slots": 0
    }
  },
  
  "measurements": {
    "throughput": {
      "tokens_per_second": 245.3,
      "requests_per_second": 3.8
    },
    "latency": {
      "time_to_first_token_ms_p50": 45,
      "time_to_first_token_ms_p99": 120,
      "end_to_end_latency_ms_p50": 2100,
      "end_to_end_latency_ms_p99": 3400,
      "inter_token_latency_ms": 15.2
    },
    "memory": {
      "gpu_memory_util_pct": 87.5,
      "allocated_gpu_memory_gb": 70.0,
      "peak_gpu_memory_gb": 72.5,
      "kv_cache_memory_gb": 32.0,
      "model_weights_gb": 35.0
    },
    "efficiency": {
      "tokens_per_second_per_gpu": 30.66,
      "tokens_per_gpu_memory_gb": 3.5,
      "effective_batch_size": 4.2
    }
  },
  
  "reliability": {
    "success": true,
    "num_errors": 0,
    "error_messages": [],
    "num_timeouts": 0,
    "num_oom": 0,
    "num_invalid_generations": 0,
    "variance": {
      "throughput_std_dev_pct": 2.1,
      "latency_p99_std_dev_pct": 5.3
    }
  },
  
  "labels": {
    "within_tested_region": true,
    "region_reasoning": {
      "out_of_region_reason": null,
      "efficiency_margin": 0.89,
      "curvature_risk": "low",
      "latency_stability": "stable"
    },
    "confidence": 0.92
  },
  
  "notes": "Stable performance. Throughput plateaus from concurrency=8 onwards. No memory pressure."
}
```

### Key Dataset Dimensions

#### Model Characteristics (Categorical + Continuous)
- Model name (categorical: llama-70b, llama-13b, mistral-7b, etc.)
- Context length (continuous: 1k–100k tokens)
- Architecture features (n_layers, hidden_dim, n_heads, n_kv_heads)

#### Hardware Configuration (Categorical + Continuous)
- GPU type (H100, A100, L40S, L4, etc.)
- GPU count (1–16+)
- GPU memory (40–80GB)
- Interconnect type (NVLink, PCIe, Ethernet)
- System memory

#### Backend Version & Flags (Categorical)
- Backend name (vLLM, TensorRT-LLM, llama.cpp, etc.)
- Backend version (semantic versioning)
- Compilation flags (flash_attn, cudnn_benchmark, etc.)
- Data type (float16, bfloat16, int8)

#### Parallelism Configuration (Continuous/Categorical)
- Tensor parallelism size: [1, 2, 4, 8, 16, ...]
- Pipeline parallelism size: [1, 2, 4, 8, ...]
- Expert parallelism size (for MoE): [1, 2, 4, ...]
- Distributed executor backend (Ray, PyTorch, etc.)

#### Sequence Length Parameters (Continuous)
- Input sequence length (min): [128, 256, 512, 1k, 2k, 4k, 8k, ...]
- Input sequence length (max): [512, 2k, 4k, 8k, 16k, 32k, 64k, ...]
- Output sequence length: [1, 32, 64, 128, 256, 512, 1k, ...]
- Sliding window size (for attention): [None, 128, 256, ...]

#### Quantization Strategy (Categorical)
- Weight quantization: [None, int8, int4, nf4, fp8, ...]
- KV cache quantization: [None, int8, int4, fp8, ...]
- Activation quantization: [None, int8, fp8, ...]

#### Concurrency & Batching (Continuous)
- Max concurrent sequences: [1, 2, 4, 8, 16, 32, 64, ...]
- Max model length (per request): [512, 1k, 4k, 8k, 16k, ...]
- Prefill chunk size: [1k, 2k, 4k, 8k, ...]
- Enable chunked prefill: [true, false]

#### Cache & Memory Management (Continuous/Categorical)
- Block size (KB): [1, 4, 8, 16, 32]
- GPU memory utilization target: [0.7, 0.8, 0.9, 0.95]
- CPU offload (GB): [0, 10, 50, 100, ...]
- Enable prefix caching: [true, false]

#### Speculative Decoding (Optional, Categorical/Continuous)
- Enable speculative decoding: [true, false]
- Number of speculative tokens: [1, 4, 8, 16, ...]

---

## Machine Learning Classifier: Decision Trees with Conditional Upgrade to XGBoost/CatBoost

### Progressive Model Selection Strategy

We use a **staged approach** starting with Decision Trees for simplicity, interpretability, and maintainability, upgrading to ensemble methods (XGBoost, CatBoost) only if Decision Trees underfit.

#### Stage 1: Decision Tree Classifier (Primary Model)

**Why Decision Trees first:**
- **Interpretable**: Rules are human-readable; supports explanation to users ("your config hits this decision rule")
- **Fast inference**: O(log n) prediction time; suitable for real-time UI validation
- **No hyperparameter tuning burden**: Tree depth is the main control; easy to validate
- **Extrapolation resistance**: Trees don't extrapolate into sparse regions—they stop at learned boundaries
- **Direct rule extraction**: Rules can be exported and hardcoded if needed

**Training configuration**:
```python
DecisionTreeClassifier(
  max_depth=6-8,              # Controls model complexity; deeper = more interaction capture
  min_samples_split=5,         # Prevent overfitting on small clusters
  min_samples_leaf=3,          # Ensure rules apply to multiple examples
  class_weight='balanced',     # Handle class imbalance (e.g., 80% within, 20% outside region)
  random_state=42
)
```

**Validation**:
- Stratified 5-fold cross-validation on the 200–300 run dataset
- Target metrics:
  - Precision (Pr(config outside region is actually outside)): ≥ 0.85
  - Recall (Pr(config outside region detected)): ≥ 0.75
  - ROC-AUC: ≥ 0.90

**Interpretability outputs**:
```python
from sklearn.tree import export_text

rules_text = export_text(tree, feature_names=feature_list)
# Example:
# |--- tensor_parallel_size > 4
# |   |--- concurrency > 24
# |   |   |--- class: outside_region (samples=8)
# |   |--- concurrency <= 24
# |   |   |--- class: within_region (samples=32)
# |--- tensor_parallel_size <= 4
# |   |--- class: within_region (samples=160)
```

---

#### Stage 2: Ensemble Models (XGBoost/CatBoost) – Conditional Upgrade

**Condition for upgrade**: If Decision Tree cross-validation performance is insufficient:
- ROC-AUC < 0.88
- Precision < 0.80 or Recall < 0.70
- Or if testing additional model/hardware/backend combos reveals consistent underfitting

**When to upgrade**:
1. Decision Trees cannot capture high-order parameter interactions (e.g., non-monotonic relationships)
2. Dataset is large enough (>300 samples) to support ensemble without overfitting
3. Computational cost is acceptable (training time <10 sec, inference time <5ms)

**XGBoost Configuration** (if needed):
```python
XGBClassifier(
  max_depth=5-7,              # Trees in ensemble are shallower than Decision Trees
  learning_rate=0.05-0.1,
  n_estimators=200-500,       # Tune with early stopping on validation set
  scale_pos_weight=(n_outside / n_within),  # Handle class imbalance
  subsample=0.8,
  colsample_bytree=0.8,
  early_stopping_rounds=20    # Stop if validation performance plateaus
)
```

**CatBoost Configuration** (alternative if categorical features dominate):
```python
CatBoostClassifier(
  depth=5-7,
  learning_rate=0.05-0.1,
  iterations=200-500,
  l2_leaf_reg=3,              # Regularization
  task_type='CPU',
  verbose=False,
  auto_class_weights='balanced'
)
```

**Why CatBoost over XGBoost for this domain**:
- Native categorical feature support (GPU type, backend name, quantization method)
- No manual encoding needed
- Better performance on mixed categorical/continuous data

**Validation**:
- Same stratified 5-fold CV setup as Decision Trees
- Ensure no degradation in precision/recall trade-off
- Check feature importance to verify learned patterns align with domain knowledge

---

#### Feature Importance & Explanation

For both Decision Trees and ensemble models:

**Decision Tree**:
- Extract rules directly via `export_text()`
- Report feature importance (built-in gini importance)

**XGBoost/CatBoost** (if upgraded):
- Use SHAP values for per-prediction explanations
- Example: "Config is outside tested region because: TP=8 (SHAP=+0.35) + Concurrency=32 (SHAP=+0.28) together create parameter interactions not covered in testing"

---

### Decision Logic for User-Facing Validation

```python
def validate_config(user_config, model_tree_or_ensemble):
  """
  Predict whether user config is within tested performance region.
  
  Args:
    user_config: dict of parameter values
    model_tree_or_ensemble: Trained Decision Tree or XGBoost/CatBoost model
    
  Returns:
    {
      'prediction': 'within_region' | 'uncertain' | 'outside_region',
      'probability': float in [0, 1],
      'explanation': str,
      'decision_rules': [str, ...]  # For decision trees
    }
  """
  
  # Encode config as feature vector
  X = encode(user_config)
  
  # Get probability of being within tested region
  proba = model_tree_or_ensemble.predict_proba(X)[0]  # [Pr(outside), Pr(within)]
  pr_within = proba[1]
  
  # Decision thresholds
  if pr_within > 0.70:
    prediction = 'within_region'
  elif pr_within > 0.50:
    prediction = 'uncertain'
  else:
    prediction = 'outside_region'
  
  # Generate explanation
  if isinstance(model, DecisionTree):
    rules = extract_path_rules(model, X)  # Returns decision path as human-readable rules
    explanation = f"Your config matches these tested characteristics: {' AND '.join(rules)}"
  else:
    shap_values = explain_with_shap(model, X)
    top_features = shap_values.argsort()[-3:]  # Top 3 contributing features
    explanation = f"Config is outside tested region due to: {', '.join(top_features)}"
  
  return {
    'prediction': prediction,
    'probability': pr_within,
    'explanation': explanation,
    'decision_rules': rules if isinstance(model, DecisionTree) else None
  }
```

---

### Deployment & Monitoring

#### Model Registry

```
models/
  llama70b-8xh100-vllm062/
    model.pkl                 # Trained Decision Tree or XGBoost
    metadata.json             # Training date, performance metrics, feature names
    rules.txt                 # Human-readable rules (Decision Tree only)
    feature_importance.json   # Feature importance scores
  
  llama70b-8xa100-vllm062/
    model.pkl
    metadata.json
    ...
```

#### Retraining Trigger

Retrain the model when:
1. New testing data is added (e.g., new model/hardware/backend combo)
2. Backend version changes (e.g., vLLM 0.6.2 → 0.7.0)
3. Cross-validation performance degrades (e.g., ROC-AUC drops below 0.85)
4. User feedback indicates false positives/negatives (gather via UI)

---

## Generalization & Scalability

### Extending to Multiple Models/Hardware/Backend Combinations

Repeat the entire pipeline (Phases 1–3, knee-detection, model training) for each unique triple (model, hardware, backend). Store models in a registry (see Deployment & Monitoring section above).

At inference time, identify the user's model/hardware/backend, load the corresponding model, and classify.

### Handling New Models/Hardware/Backend

Once a new combination appears in testing:
1. Run Phases 1–3 (200–300 runs, ~5–10 hours)
2. Run knee-detection algorithm (automated)
3. Train Decision Tree classifier (< 1 second)
4. Validate performance; upgrade to XGBoost/CatBoost only if needed
5. Deploy new model to registry

### Transfer Learning (Future Work)

If a new model is very similar to an existing one (e.g., Llama-70B vs Llama-70B-Instruct):
1. Initialize a Decision Tree with rules from the closest existing model
2. Fine-tune on a subset of new data (50–100 runs)
3. This reduces data requirements for model variants

---

## Application: User-Facing Validation

### Integration with Configuration Recommendations

When a user configures an inference system:

1. **Show the recommendation** (e.g., "For LLaMA-70B on 8×H100 with 8k context, we recommend TP=4, concurrency=16")
2. **Validate against tested region**: Input user config into trained classifier
3. **Display confidence**:
   - **Green ("Within Tested Region")**: Pr(within) > 0.70 — directly supported by testing and classification
   - **Yellow ("Uncertain")**: 0.50 < Pr(within) ≤ 0.70 — within parameter ranges but near boundaries or interaction edges
   - **Red ("Outside Tested Region")**: Pr(within) < 0.50 — outside tested performance boundaries per classifier
4. **Provide actionable feedback**: 
   - For Decision Trees: "Your config violates these tested characteristics: [list extracted rules]"
   - For ensembles: "Your config is outside tested region because: tensor_parallel_size=8 + concurrency=32 is uncommon and not validated in our testing"

### Example UI Integration

```
Model: LLaMA-70B | Hardware: 8×H100 | Backend: vLLM 0.6.2
┌─────────────────────────────────────────┐
│ Recommended Configuration                │
├─────────────────────────────────────────┤
│ Tensor Parallelism: [4] (6 other opts)  │
│ Pipeline Parallelism: [2] (3 other opts)│
│ Input Seq Len: [8192] (5 other opts)    │
│ Concurrency: [16] (6 other opts)        │
├─────────────────────────────────────────┤
│ ✓ WITHIN TESTED REGION (Pr=0.89)       │
│                                         │
│ This configuration has been tested      │
│ extensively and delivers stable         │
│ performance.                            │
│                                         │
│ Expected: 245 tok/s, 87% GPU util      │
└─────────────────────────────────────────┘

[User adjusts: TP → 8, Concurrency → 32]

┌─────────────────────────────────────────┐
│ ⚠ OUTSIDE TESTED REGION (Pr=0.38)      │
│                                         │
│ Your configuration is not covered by    │
│ our empirical testing:                  │
│                                         │
│ Decision Tree Rules:                    │
│ • IF TP > 4 AND Concurrency > 24      │
│   THEN outside region (8 test cases)   │
│ • IF TP >= 8: not validated            │
│                                         │
│ Recommendations:                        │
│ • Reduce Concurrency to 24              │
│ • Keep TP=4 (tested extensively)       │
│ • Increase GPU count if more capacity   │
└─────────────────────────────────────────┘
```

---

## Comparative Advantages

### vs. Conservative "Tested Only" Approach
- **Problem solved**: Rejects valid configurations due to sparsity
- **Our solution**: Validates within tested performance boundaries discovered by testing, not just exact matches
- **Result**: 5–10× more configurations validated while maintaining boundary safety

### vs. "Rule-Based" Thresholds
- **Problem solved**: Hard-coded rules (e.g., "TP ≤ 64") miss parameter interactions
- **Our solution**: Data-driven via Decision Trees or ensembles; captures multi-parameter interactions automatically
- **Result**: More accurate, adapts as backends evolve

### vs. Naive "Fit Any Surrogate Model"
- **Problem solved**: Sparse testing data causes extrapolation failure in untested regions
- **Our solution**: Conservative knee-detection + margin ensures we only classify within tested regions; Decision Trees resist extrapolation
- **Result**: High precision; avoids confident predictions in sparse areas

### vs. Black-Box Complex Ensemble (No Staging)
- **Problem solved**: Users and support staff cannot understand why a config is outside region
- **Our solution**: Start with interpretable Decision Trees; only upgrade to ensembles if insufficient accuracy
- **Result**: Maintainability, explainability, faster iteration

---

## Patent Considerations

### Inventive Concepts

1. **Adaptive Fractional Factorial Testing for High-Dimensional Configuration Spaces**
   - Novelty: Using statistical design-of-experiments (FFD) to systematically cover a high-dimensional parameter space, followed by adaptive refinement
   - Applicable to any system where configuration testing is expensive and parameter interactions are non-linear

2. **Automated Knee-Detection via Curvature & Efficiency Metrics**
   - Novelty: Multi-criterion algorithm that detects performance knees in benchmark data without hand-coded thresholds
   - Generalizable to any performance prediction problem with noisy, non-linear relationships

3. **Progressive Classification for Operating Region Boundary Prediction**
   - Novelty: Training an interpretable classifier (Decision Trees) first, then conditionally upgrading to ensemble methods only if underfitting; labeling based on empirically-discovered performance region boundaries
   - Applicable to any system where operating regions must be inferred from sparse testing and explainability is valued

### Scope for Patent Claims

**Apparatus claim** (broad):
> A system for validating configurations of a machine learning inference system comprising: (a) a testing engine that executes a fractional factorial design to systematically sample a configuration parameter space; (b) an adaptive refinement module that identifies high-curvature regions and densifies sampling accordingly; (c) a knee-detection engine that labels sampled configurations as within or outside tested performance regions based on performance metrics and curvature; (d) a classifier training module that trains an initial Decision Tree classifier and conditionally upgrades to ensemble methods if underfitting is detected; (e) an inference engine that predicts whether user-provided configurations are within tested performance regions and provides interpretable explanations.

**Method claim** (process-focused):
> A method for predicting whether configurations operate within tested performance regions comprising: (1) generating a fractional factorial design for a configuration parameter space; (2) executing benchmarks for each design point; (3) computing performance curvature and efficiency metrics; (4) labeling configurations as within or outside tested regions based on knee-detection criteria; (5) training an initial Decision Tree classifier and validating performance; (6) if ROC-AUC < threshold, training ensemble models (XGBoost/CatBoost) as an upgrade; (7) applying the selected classifier to user configurations to predict region membership and provide explanations.

### Prior Art Considerations

- Fractional factorial design: Well-known in engineering (Box, Hunter, Hunter)
- Decision Tree classification: Standard since 1980s
- XGBoost: Published 2016
- **Novel combination**: Applying FFD + adaptive sampling + automated knee-detection + progressive Decision Tree/ensemble classification to the specific problem of LLM inference configuration validation

Differentiation: No prior work combines these techniques for LLM inference configuration, and the progressive model-selection strategy (DT first, upgrade conditionally) is novel for configuration validation.

---

## Implementation Roadmap

### Phase A: Proof of Concept
1. Implement knee-detection algorithm (pseudocode → Python)
2. Conduct Phase 1 testing on a single (model, hardware, backend) triple
3. Generate labels via knee-detection
4. Train Decision Tree classifier
5. Validate on held-out test set; assess if upgrade to XGBoost is needed

### Phase B: Scale to Production
1. Implement adaptive sampling for Phases 2–3
2. Test on 3–5 model/hardware/backend combinations
3. Build model registry and inference service
4. Upgrade models to XGBoost/CatBoost only for combos where DT underfits
5. Integrate into ConfigIQ UI

### Phase C: User Validation & Iteration
1. Gather feedback on confidence levels (green/yellow/red)
2. Adjust knee-detection thresholds if needed
3. Monitor false positive/negative rates; retrain as needed
4. Refine Decision Tree depth and ensemble hyperparameters
5. Deploy to production

---

## Conclusion

This approach transforms LLM inference configuration validation from a binary "tested/untested" decision into a nuanced, data-driven prediction of whether a configuration operates within tested performance regions. By combining statistical design-of-experiments with automated performance analysis and progressive machine learning (starting simple with interpretable Decision Trees, upgrading only if needed), we enable users to explore configurations confidently while maintaining boundaries backed by empirical evidence.

The system is:
- **Efficient**: 200–300 runs per model/hardware/backend vs. exhaustive search
- **Automated**: No hand-coded rules; algorithms adapt to data
- **Interpretable**: Decision Trees provide human-readable rules; ensemble upgrades include SHAP explanations
- **Maintainable**: Starting with Decision Trees reduces complexity; ensemble upgrades are conditional, not default
- **Scalable**: Extends to new models, hardware, backends trivially
- **Generalizable**: Techniques apply to any configuration-optimization problem with expensive evaluation

---

## References & Further Reading

- Box, Hunter, Hunter (1978). *Statistics for Experimenters*. Foundational reference on fractional factorial design.
- Breiman, L. (1984). *Classification and Regression Trees*. Decision tree theory and practice.
- Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16.* XGBoost foundational paper.
- Dorogush et al. (2018). "CatBoost: Gradient Boosting for Categorical Features." *arXiv:1810.11372.* CatBoost for mixed feature types.
- Bergstra & Bengio (2012). "Random Search for Hyperparameter Optimization." *JMLR.* — Motivation for adaptive vs. fixed sampling.
- SHAP (Lundberg & Lee, 2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS '17.* For interpretability of ensemble models.

