# Validating LLM Inference Configurations Against Real-World Testing

## Overview

This document explains how we can validate user configurations against actual test data, so we can tell them "your setup will work" or "your setup might hit problems" with confidence.

**The Challenge**: LLM inference has many knobs to turn. A user can tweak tensor parallelism, batch size, sequence length, quantization, and a dozen other settings. Each combination behaves differently. Some work great. Some hit performance cliffs where throughput drops sharply. Some run out of memory. We need to know which is which.

**What We're Doing**: We're going to run a large number of structured tests across different configurations, automatically detect where performance breaks down, then train a simple machine learning model to predict whether a user's configuration will work or not.

---

## The Problem We're Solving

### Today's Approach is Too Simple

Right now, we have a binary flag: "tested" or "untested". But that's not nuanced enough:
- A configuration might be "untested" but still work fine (it's within the safe zone we haven't explicitly benchmarked)
- A configuration might be "theoretically valid" but perform terribly (it's on the other side of a performance cliff)

Users get confused: "Why is my config flagged as untested when it looks reasonable?" and support has no good answer.

### Why It's Hard to Solve

1. **Many parameters**: We're talking 8–15 different settings per model/hardware/backend combo
2. **Parameters interact**: Changing one setting changes how another behaves. It's not independent.
3. **Performance cliffs are hidden**: There's a sweet zone where throughput is good, but move one parameter outside that zone and suddenly you lose 40% performance or hit OOM
4. **Testing is expensive**: Each configuration needs GPU time to benchmark properly
5. **You can't guess**: Simple rules like "don't use TP > 64" miss all the interactions and don't adapt when backends change

---

## Approach: Structured Testing + Automated Boundary Detection + Learned Models

1. **Run adaptive tests** designed to efficiently cover the parameter space
2. **Automatically detect performance boundaries** (fully automated; no manual tuning required)
3. **Label each tested configuration** as within or outside tested performance regions
4. **Train an interpretable model** (decision trees first; ensembles only if needed for accuracy)
5. **Use the model to classify new user configurations**

### Phase 1: Initial Broad Coverage (~90–150 benchmarks)

Use a statistical technique called "fractional factorial design" (FFD). Instead of testing every combination (which would be 200+ runs), we test maybe 1 in 3 combinations, chosen strategically. This gives us:
- A good sense of which parameters matter
- First signs of where things break down
- Enough data to move to Phase 2

**Why this matters**: It's the industrial engineering gold standard for exploring complex systems when testing is expensive.

### Phase 2: Zoom In on Problem Zones (~50–100 benchmarks)

We look at the Phase 1 results and ask: "Where does performance drop off sharply?" Then we:
- Add more tests around those zones to find the exact cliff edge
- Test parameter combinations that interact (e.g., does increasing tensor parallelism break when you also increase batch size?)
- Binary-search for OOM boundaries

**Why this matters**: We get precise knowledge of where the danger zones are, not just hints.

### Phase 3: Confirm Boundaries (~20–50 benchmarks)

We test configurations right at the edges we discovered, to make sure:
- The boundaries are real and repeatable
- We understand how close we can safely get to them

**Total investment**: ~200–300 benchmarks per (model, hardware, backend) combo. That's 5–10 hours of GPU time, typically parallelizable across multiple machines.

---

## Automated Boundary Detection

After testing, each configuration is automatically labeled `within_tested_region: true/false` using a multi-criterion algorithm. No manual review or tuning is required—all thresholds are computed from the dataset itself and adapt automatically to different hardware and models.

### Detection Mechanism: Adaptive, Cohort-Relative Thresholds

The algorithm uses five independent checks. Each relies on relative metrics (percentiles, z-scores, ratios) that automatically calibrate to the dataset:

**Check 1: Obvious Failures**
- Configuration crashed, timed out, or ran out of memory → Outside region
- GPU memory utilization >95% → Outside region (too close to OOM boundary)

**Check 2: Performance Knee Detection**
Detects curvature in the throughput landscape. For each parameter, we compute how sharply performance changes as you vary that parameter. High curvature indicates proximity to a knee (sharp performance dropoff).

- Compute second derivative of throughput relative to each parameter
- Compare against 75th percentile of curvatures in dataset
- If a configuration has above-median curvature → likely near a knee → flag as outside region

This automatically adapts: hardware with steep throughput cliffs has higher absolute curvatures, but uses the same percentile threshold.

**Check 3: Efficiency vs. Cohort (Efficiency Cliff Detection)**
Compares each configuration's efficiency (tokens/sec per GPU) to the distribution of all tested configs.

- Compute z-score: `(config_efficiency - median_efficiency) / std_dev`
- If z-score < -1.5 (more than 1.5 standard deviations below median) → outside region

Automatically adapts: hardware with stable efficiency has tight variance (narrow band); hardware with variable efficiency has looser variance (wider band). Both use the same 1.5 std threshold.

**Check 4: Latency Stability**
Even if throughput is acceptable, pathological latency spikes (queueing, GC, resource contention) indicate an unstable configuration.

- Establish baseline: median p99 latency among high-throughput, stable configs
- If configuration's p99 latency > 3× baseline → outside region

Automatically adapts: hardware with inherently higher latency has higher absolute threshold, but same 3× margin.

**Check 5: Conservative Safety Margin**
Require configurations to stay at least 15% away from where efficiency starts declining. Provides buffer for variance and incomplete test coverage.

- If `efficiency < (0.85 × cohort_median_efficiency)` → outside region

### Why This Works Across All Hardware/Models

All thresholds are **relative metrics**: percentiles, z-scores, multiples of baseline. A V100 and H100 have vastly different throughput ranges and latency characteristics, but both have knees, efficiency cliffs, and latency anomalies. The algorithm detects these using cohort-relative signals, so it requires no recalibration when testing a new model, hardware, or backend version.

---

## The Data We Collect

For each benchmark, we record:

**Configuration Settings**:
- Model (name, size, architecture details)
- Hardware (GPU type, count, memory, interconnect)
- Backend (vLLM, TensorRT-LLM, etc., plus version)
- Parallelism: tensor size, pipeline size, etc.
- Sequence lengths: min/max input, output length
- Quantization: weight, KV cache, etc.
- Batch settings: max concurrent requests, prefill chunk size
- Memory tuning: cache block size, GPU memory target, prefilling strategy
- Optional: speculative decoding settings

**What We Measure**:
- Throughput (tokens/second, requests/second)
- Latency (time to first token, end-to-end, p50 and p99)
- Memory (GPU utilization %, peak usage)
- Efficiency (tok/s per GPU, tok/s per GB)
- Whether it ran successfully or failed

**Our Label**:
- `within_tested_region: true/false` (from the cliff-detection algorithm)
- Why we labeled it that way
- Confidence level

---

## The Model: Simple First, Fancy Later

### Primary: Decision Trees

We start with decision trees because they're:
- **Explainable**: Rules are English-like. "IF tensor_parallel > 4 AND concurrency > 24 THEN outside region"
- **Fast**: Prediction takes microseconds; good for UI validation
- **Resistant to extrapolation**: They don't guess in regions they haven't seen
- **Easy to maintain**: No mysterious hyperparameter tuning

Configuration:
```
max_depth: 6–8        (controls how many rules deep)
min_samples per leaf: 3 (each rule must apply to real data)
balanced classes      (we handle the case where most configs are fine)
```

We validate this with 5-fold cross-validation on our test data. Target: 90%+ accuracy, 85%+ precision, 75%+ recall.

### Backup: Ensemble Methods (Only If Needed)

If decision trees aren't accurate enough (ROC-AUC < 0.88), we upgrade to:
- **XGBoost** or **CatBoost** for more complex interactions

But we only do this if the simpler approach isn't working. Fancier models are harder to explain to users and harder to maintain.

---

## How We Use This

### In ConfigIQ

User configures an inference setup → we show them our recommendation → they tweak it slightly → we check their tweaked config against the model:

```
Recommended: TP=4, concurrency=16, batch_size=32
User's choice: TP=8, concurrency=32, batch_size=64

✓ GREEN (90% confidence): "Within tested region. Expect 245 tok/s."
⚠ YELLOW (60% confidence): "Uncertain—near untested boundaries. Might work, might not."
✗ RED (20% confidence): "Outside tested region. We haven't seen this work. Try TP=4 instead."
```

For green: "This configuration has been thoroughly tested and delivers stable performance."

For yellow: "Your config is on the edge of what we've tested. It might work, but watch your latency metrics in production."

For red: "Your config combines parameters we haven't validated together. Here are safer alternatives..."

### Why This Matters

- **Users get confidence**: They know whether they're in safe territory
- **Support gets answers**: We can say "your config hits decision rule #47, which we found problematic in testing"
- **We iterate faster**: As we test more backends/models, the validation automatically gets better
- **We avoid bad recommendations**: We don't confidently recommend configs that will fail

---

## Workflow: Adding a New Model/Hardware/Backend

When we want to support a new combo (e.g., Llama-70B on 8×A100 with vLLM 0.7):

1. Run Phases 1–3 of testing (~5–10 hours, ~200–300 benchmarks)
2. Run the cliff-detection algorithm (automatic, ~1 minute)
3. Train a decision tree (automatic, <1 second)
4. Check validation metrics; if decision tree isn't accurate enough, train XGBoost
5. Add model to our registry; done

Next time a user tries that combo, we validate against real testing data.

---

## Compared to Other Approaches

### vs. "Just Mark Exact Matches as Tested"
- **Problem**: Rejects tons of valid configs because we didn't test that exact combination
- **Our approach**: Validates the zone around tested configs
- **Win**: 5–10× more configs validated, same safety level

### vs. "Use Hand-Coded Rules"
- **Problem**: Rules like "TP ≤ 64" miss interactions and become wrong as software changes
- **Our approach**: Data-driven; learns from actual benchmarks
- **Win**: More accurate, adapts automatically

### vs. "Fit a Neural Net to Everything"
- **Problem**: Sparse data means the model guesses in untested regions. Your users get confident bad answers.
- **Our approach**: Conservative cliff-detection ensures we only validate regions we've actually tested; simple models resist extrapolation
- **Win**: High precision; we admit uncertainty when appropriate

---

## Expanding Over Time

We don't need to test all combos from day one.

**Phase 1**: Test 3–5 important combos (e.g., popular models on popular hardware). Get the process working.

**Phase 2**: Users ask "what about X?". We test X. Validation improves.

**Phase 3**: We've built a database of tested combos. New tests fill gaps. The system becomes more and more useful.

---

## What Success Looks Like

- Users get green/yellow/red confidence signals
- Support can explain flagged configs: "your TP is fine, but TP + concurrency combo is untested"
- Recommendations avoid performance cliffs
- As we test more, validation coverage improves automatically
- No false positives: "safe" configs actually work in production

---

## Open Questions for the Team

1. **Testing capacity**: How many benchmark runs can we do per week? That determines how fast we can expand coverage.
2. **Which combos first?**: Which model/hardware/backend combos should we test first to maximize user impact?
3. **Confidence thresholds**: What probabilities do we use for green/yellow/red? (e.g., >0.70 = green)?
4. **Handling variance**: If a config sometimes OOMs and sometimes works, how do we label it? (Probably "outside region" to be safe.)
5. **Feedback loop**: Do we gather user data from production to detect when our predictions were wrong? If so, how?

---

## Technical Setup (For Engineers)

### Data Pipeline

```
Benchmark runs (guidellm)
  ↓
Parse results → JSON per run
  ↓
Cliff-detection algorithm
  ↓
Labeled dataset (within_tested_region: true/false)
  ↓
Train decision tree (sklearn)
  ↓
Validate on holdout set
  ↓
[If good: ship. If bad: train XGBoost]
  ↓
Serialize model → registry
  ↓
Load in ConfigIQ API at validation time
```

### Model Files

```
models/
  llama70b-8xh100-vllm062/
    model.pkl              # Trained decision tree
    metadata.json          # Training date, accuracy, features
    rules.txt              # Human-readable rules
  
  llama70b-8xa100-vllm062/
    model.pkl
    ...
```

### Integration Point

In ConfigIQ backend (validation endpoint):

```python
def validate_config(model_name, hardware, backend, config_dict):
  model = load_model(model_name, hardware, backend)
  features = encode(config_dict)
  prob_within = model.predict_proba(features)
  
  if prob_within > 0.70:
    return {'status': 'within_region', 'confidence': prob_within}
  elif prob_within > 0.50:
    return {'status': 'uncertain', 'confidence': prob_within}
  else:
    return {'status': 'outside_region', 'confidence': prob_within}
```

---

## Why This Approach?

**Efficiency**: Fractional factorial + adaptive sampling cuts testing time by 70% vs. naive grid search.

**Accuracy**: Decision trees are simple but capture parameter interactions. XGBoost available if we need it.

**Explainability**: Users and support staff understand why a config is flagged.

**Sustainability**: No hand-coded rules to maintain. As backends change, we just retest and retrain.

**Scalability**: Add a new model/hardware/backend combo in a day. System automatically covers more ground.

---

## Next Steps

1. **Pick first combo to test**: Which model/hardware/backend has highest demand?
2. **Set up test harness**: Phases 1–3 infrastructure
3. **Build cliff-detection**: Implement the algorithm
4. **Collect first dataset**: ~200–300 benchmarks
5. **Train & validate**: Decision tree + cross-validation
6. **Integrate**: Validation endpoint in ConfigIQ
7. **Monitor**: Track false positives/negatives; iterate

---

## Questions?

This is a team discussion doc. What's unclear? What would help? What are your concerns?

