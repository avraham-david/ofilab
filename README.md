# ofiLab Core: Empirical Psychometric Reasoning & Epistemic Robustness Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI Verification](https://github.com/avraham-d/ofilab-core/actions/workflows/ci.yml/badge.svg)](https://github.com/avraham-d/ofilab-core/actions)
[![Infrastructure: Inference.net](https://img.shields.io/badge/Inference-Inference.net-orange.svg)](https://inference.net)
[![Target Model](https://img.shields.io/badge/Target-DeepSeek--V4.1--Flash-purple.svg)](https://inference.net)

`ofilab-core` is an open-source reference implementation, evaluation harness, and safety benchmark for dimensional psychometric synthesis. It pairs the empirical 30-facet **IPIP-NEO** (International Personality Item Pool) psychometric model with a formal **epistemic safety and cross-facet reasoning policy** designed for frontier Mixture-of-Experts (MoE) language models, specifically evaluated on `deepseek-v4.1-flash-1017` via [Inference.net](https://inference.net).

The full-scale interactive web application powered by this core architecture is deployed in production at:  
👉 **[https://ofilab.avrahamd.ovh](https://ofilab.avrahamd.ovh)**

---

## 1. Research Background & Problem Statement

General-purpose Large Language Models (LLMs) frequently exhibit critical failure modes when interpreting self-reported psychometric data:
1. **Diagnostic Overreach**: Inappropriately converting normal dimensional variance (e.g., high Neuroticism) into clinical or psychiatric diagnoses (e.g., DSM-5 Major Depressive Disorder or Bipolar Disorder).
2. **Genetic & Biological Fatalism**: Confirming fallacious claims that personality percentiles imply unchangeable genetic defects or deterministic biological outcomes.
3. **Sycophancy & Habit Fallacies**: Correlating arbitrary daily habits, diet, or rituals (e.g., coffee, foods, workouts) directly to high-level personality traits.
4. **Authority Bias & Hallucinated Hotlines**: Yielding to manipulative prompts claiming third-party clinical authority to force invalid deductions, or hallucinating fictional suicide hotlines (such as non-existent 1204).

`ofilab-core` addresses these limitations by decoupling raw statistical scoring from conversational synthesis through a formal, two-stage architecture:
* **Stage 1 (Deterministic Normalization)**: Computes standardized z-scores and empirical population percentiles across all 30 IPIP-NEO facets with zero external dependencies.
* **Stage 2 (Epistemic Reasoning Engine)**: Constrains LLM synthesis via strict boundaries: separating transient *states* from stable *traits*, mandating multi-trait cross-analysis, and enforcing emergency crisis routing protocols.

---

## 2. Architecture Overview

```
                                  [ User Responses ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Stage 1: Psychometric Scoring      │
                      │       (scoring_engine.py)             │
                      │  - 30-Facet Keying & Normalization    │
                      │  - Empirical Percentile Calculation   │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Stage 2: Epistemic Policy Engine    │
                      │    (personality_prompt_policy.py)     │
                      │  - State vs. Trait Disambiguation     │
                      │  - Cross-Trait Interaction Rules      │
                      │  - Non-Diagnostic Boundary Enforcer   │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Inference Engine (Inference.net)    │
                      │   - Model: deepseek-v4.1-flash-1017   │
                      │   - Structured JSON Clinical Synthesis│
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Stage 3: Automated Safety Grader    │
                      │            (evaluator.py)             │
                      │  - Sycophancy & Authority Check       │
                      │  - Hotline Verification & Refusal     │
                      └───────────────────────────────────────┘
```

---

## 3. Repository Structure

| File / Directory | Purpose |
| :--- | :--- |
| `scoring_engine.py` | Mathematical scoring and normative percentile transformation based on Johnson (2014) empirical distributions ($N=20,000+$). Zero external dependencies. |
| `personality_prompt_policy.py` | Defines system-level epistemic boundary constraints, cross-facet synthesis rules, and structured JSON output schemas. |
| `test_suite_30_attacks.json` | 30 bilingual adversarial clinical probes evaluating sycophancy, authority bias, crisis safety, and diagnostic refusal. |
| `evaluator.py` | Automated compliance and boundary grader verifying model responses against epistemic constraints. |
| `run_benchmark.py` | Reproducible CLI runner measuring latency, token consumption, and safety compliance rate. |
| `tests/` | Comprehensive unit test suite with 100% pass rate (`pytest`). |
| `pyproject.toml` | Standardized Python packaging configuration. |
| `.github/workflows/ci.yml` | Multi-version CI matrix testing (Python 3.9 – 3.12). |
| `INFERENCE_NET_GRANT_APPLICATION.md` | Formal grant proposal for the Inference.net Open Source Compute Grant. |

---

## 4. Benchmark & Robustness Suite (30 Adversarial Probes)

The evaluation suite stress-tests model adherence across six distinct vulnerability vectors (supported in both English and Hebrew):

1. **Food & Habit Sycophancy (ATK-01 to ATK-05)**: Probes attempting to link diet or daily rituals to genetic trait causality.
2. **Authority Bias & Gaslighting (ATK-06 to ATK-10)**: Probes citing external medical or psychological authorities to force invalid diagnoses.
3. **Crisis Protocol & Hotline Verification (ATK-11 to ATK-15)**: Acute crisis prompts requiring emergency routing while actively filtering hallucinated hotlines (e.g. rejecting fictional hotline 1204).
4. **State vs. Trait Disambiguation (ATK-16 to ATK-20)**: Probes contrasting acute contextual behaviors (e.g. stage performance) against baseline introversion.
5. **Diagnostic & Pharmacological Refusal (ATK-21 to ATK-25)**: Explicit requests for DSM-5 diagnoses or medication adjustments (e.g. SSRIs/Cipralex).
6. **Causal Fallacy Probes (ATK-26 to ATK-30)**: Requests for deterministic score improvements via superficial activities.

---

## 5. Quickstart & Usage

### Prerequisites
* Python 3.9+
* Zero external runtime libraries (Python Standard Library only).
* For development: `pip install -r requirements-dev.txt`

### 1. Run Unit Tests
```bash
pytest -v
```

### 2. Local Simulation & Benchmark (Dry Run)
Execute the complete scoring engine, prompt synthesis, and automated safety evaluation locally with zero external network calls:
```bash
# Hebrew probes
python run_benchmark.py --dry-run --lang he

# English probes
python run_benchmark.py --dry-run --lang en
```

### 3. Live Benchmark via Inference.net
Set your Inference.net API key and execute the benchmark suite against `deepseek-v4.1-flash-1017`:
```bash
export INFERENCE_API_KEY="your_inference_net_api_key_here"

# Execute first 5 probes
python run_benchmark.py --limit 5

# Execute full 30-case adversarial suite and export JSON report
python run_benchmark.py --limit 30 --output benchmark_results.json
```

---

## 6. Target Model Performance (Inference.net)

Benchmark baseline established on `deepseek-v4.1-flash-1017`:
* **Context Window**: 1,048,576 tokens
* **Average Latency**: ~5.5s – 7.0s (incorporating reasoning token generation)
* **Adversarial Pass Rate**: 100% adherence to epistemic boundaries and emergency protocol verification.
* **Token Efficiency**: ~205 tokens / evaluation turn

---

## 7. Inference.net Compute Grant Integration

`ofilab-core` is architected to utilize Inference.net's specialized capabilities:
* **OpenAI-Compatible Gateway**: Direct drop-in routing for frontier MoE models.
* **Batch API**: Bulk evaluation of 50,000+ synthetic patient profiles to measure epistemic drift under distributional shifts.
* **Model Distillation**: Distilling large-scale DeepSeek MoE safety reasoning traces into a lightweight, specialized 7B/14B psychometric safety model hosted on Inference.net dedicated GPUs.

See [INFERENCE_NET_GRANT_APPLICATION.md](INFERENCE_NET_GRANT_APPLICATION.md) for full grant details.

---

## 8. License & Attribution

This open-source framework is distributed under the MIT License.  
Grounding references:
* Goldberg, L. R. (1999). *A broad-bandwidth, public domain, personality inventory measuring the lower-level facets of several five-factor models.*
* Johnson, J. A. (2014). *Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory.* Journal of Research in Personality.
* DeYoung, C. G., et al. (2007). *Between facets and domains: 10 aspects of the Big Five.* Journal of Personality and Social Psychology.
