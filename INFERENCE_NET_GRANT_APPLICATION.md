# Inference.net Compute Grant Application Dossier

This document contains the complete, pre-formatted application text to submit to the **Inference.net Open-Source AI Grant Program** (up to $10,000 in free compute credits).

---

## 1. Application Metadata & Form Fields

* **Applicant Name**: Avraham D. (ofiLab Core Research Lead)
* **Contact Email**: avraham@ofilab.avrahamd.ovh / your-email@domain.com
* **Project Name**: ofiLab Core (`ofilab-core`)
* **Project Type**: Novel Applications & Developer Tools / Testing Frameworks
* **Open Source Repository**: `https://github.com/avraham-d/ofilab-core` (or target GitHub URL)
* **Live Production Application**: `https://ofilab.avrahamd.ovh`
* **Target Model on Inference.net**: `deepseek-v4.1-flash-1017` / DeepSeek MoE family

---

## 2. Project Description (Exact Submission Copy)

> *Copy and paste the text below into the "Project Description / How compute will help" field on the Inference.net application portal:*

### Project Overview
`ofilab-core` is an open-source empirical psychometric reasoning and AI safety evaluation framework. It bridges standard psychometric science (the 30-facet IPIP-NEO model based on Goldberg 1999 and Johnson 2014) with formal epistemic safety boundaries for frontier LLMs. 

The framework powers a live, zero-knowledge psychometrics web platform at **https://ofilab.avrahamd.ovh**. 

### The Core Problem
When consumer AI models interpret psychometric data, they suffer from severe epistemic vulnerabilities:
1. **Diagnostic Overreach**: Converting normative dimensional trait scores (e.g., high Neuroticism) into psychiatric DSM-5 diagnoses (Major Depression, Bipolar).
2. **Biological & Genetic Fatalism**: Telling users that their personality percentiles represent deterministic biological or genetic defects.
3. **Sycophancy & Habit Fallacies**: Validating arbitrary daily diets or rituals as causal modifiers of stable traits.
4. **Crisis Mishandling & Hotline Hallucination**: Failing acute suicidal safety prompts or hallucinating fictional hotlines (e.g., endorsing non-existent line 1204).

### The Architecture & Innovation
`ofilab-core` solves this via a decoupled 2-stage architecture:
* **Stage 1 (Pure Deterministic Scoring)**: Zero-dependency mathematical normalizer converting raw Likert inputs into empirical population z-scores and percentiles ($N=20,000+$ peer-reviewed normative distributions).
* **Stage 2 (Epistemic Reasoning Policy)**: A strict system-level constraint engine enforcing non-diagnostic boundaries, state-vs-trait disambiguation, cross-facet multi-trait synthesis, and validated crisis routing protocols.
* **Stage 3 (Automated Safety & Robustness Benchmark)**: A 30-probe adversarial evaluation harness with automated grading across 6 vulnerability categories, measuring latency, token usage, and safety compliance.

### Why Inference.net & Compute Resource Plan ($10,000 Compute Allocation)
We specifically targeted Inference.net as our reference infrastructure due to your high-throughput OpenAI-compatible gateway, cost-effective Batch API, and frontier MoE hosting (`deepseek-v4.1-flash-1017`).

The $10,000 compute grant will directly power three milestones:

1. **Massive-Scale MoE Epistemic Drift Benchmark (Inference.net Batch API - $3,500)**:
   - Generate and evaluate 50,000 synthetic psychometric profiles across diverse demographic normative distributions.
   - Run 1.5 million adversarial probe evaluations using Inference.net's discounted Batch API to systematically quantify sycophancy drift and hallucination rates across open-weights MoE architectures.

2. **Model Distillation & Fine-Tuning Pipeline (Inference.net Distillation Infrastructure - $4,500)**:
   - Distill high-quality DeepSeek V4.1 reasoning traces into an ultra-fast, lightweight 7B/14B dedicated "Epistemic Psychometric Guard" model.
   - Deploy the distilled model onto Inference.net dedicated GPU endpoints to power real-time, zero-latency clinical-grade safety verification for our live platform (`ofilab.avrahamd.ovh`).

3. **Public Open-Source Showcase & Ecosystem Contribution ($2,000)**:
   - Publish a comprehensive research whitepaper and technical case study demonstrating how Inference.net provides superior latency and cost efficiency for production MoE reasoning compared to centralized providers.
   - Feature Inference.net prominent badges (`Powered by Inference.net`) across our open-source repositories and live production application.

---

## 3. Video Demo Script (Optional Submission - 60 to 90 seconds)

* **[0:00 - 0:15] Hook**: "Most frontier LLMs hallucinate psychiatric diagnoses or validate dangerous medical advice when interpreting personality tests. Here is `ofilab-core`."
* **[0:15 - 0:40] Architecture**: Show the repo, the zero-dependency `scoring_engine.py`, the 30 facets of IPIP-NEO, and the live web app `https://ofilab.avrahamd.ovh`.
* **[0:40 - 1:05] Live CLI Benchmark**: Run `python run_benchmark.py --limit 30 --lang en` in the terminal, showing the real-time evaluation against `deepseek-v4.1-flash-1017` on `api.inference.net`. Highlight the 100% pass rate on adversarial probes including hotline hallucination resistance.
* **[1:05 - 1:20] Call to Action**: "With the Inference.net compute grant, we will scale this to a 50,000-profile batch evaluation and distill a specialized open-source psychometric safety model using Inference.net infrastructure. Thank you!"

---

## 4. Why This Proposal Stands Out to Inference.net Reviewers

1. **Infrastructure Consumption**: It is designed to consume hundreds of millions of batch tokens and use their distillation feature—exactly what Inference.net wants to showcase.
2. **Production Validation**: It is not a speculative idea; it already backs a functioning production application (`https://ofilab.avrahamd.ovh`).
3. **Rigorous Engineering**: MIT license, multi-version CI/CD matrix, pytest suite with 100% pass rate, zero external runtime dependencies.
4. **Direct Alignment with DeepSeek MoE**: Actively showcases their DeepSeek-V4.1-Flash endpoint.
