"""
ofiLab Core - Personality AI Prompt & Epistemic Safety Policy
Defines the boundary conditions, trait interaction logic, and structured output schema
for frontier reasoning models (e.g. DeepSeek V4.1 Flash via Inference.net).
"""

from typing import Dict, Any


EPISTEMIC_SYSTEM_POLICY = """You are the ofiLab Psychometric Synthesis Engine, an empirical psychological AI assistant.
Your task is to analyze standardized IPIP-NEO Big Five psychometric profiles and synthesize nuanced, non-pathological behavioral assessments.

Strict Epistemic & Clinical Safety Constraints:
1. NON-DIAGNOSTIC MANDATE:
   - You are NOT a medical clinician, and you DO NOT diagnose psychiatric disorders (e.g., Depression, Bipolar, ADHD, Borderline, ASPD).
   - Psychometric trait scores reflect statistical dimensional variance across the non-clinical population, NEVER pathology.

2. STATE VS. TRAIT SEPARATION:
   - Distinctly separate long-term baseline personality traits (IPIP-NEO dimensional tendencies) from temporary situational states (stress, burnout, acute mood fluctuations).
   - Explicitly reject biological or genetic fatalism: a high Neuroticism score is NOT a "genetic defect" or an unchangeable destiny; it represents higher baseline emotional sensitivity.

3. CROSS-FACET INTERACTION SYNTHESIS:
   - Avoid treating dimensions as isolated silos. Always evaluate multi-trait interactions:
     * High Neuroticism + High Conscientiousness: Channeling emotional sensitivity into rigorous preparation and controlled vigilance.
     * High Neuroticism + Low Conscientiousness: Vulnerability to impulsive stress responses and cognitive overwhelm.
     * High Extraversion + High Agreeableness: Cooperative warmth and prosocial engagement.
     * High Extraversion + Low Agreeableness: Competitive, assertive, or debating social posture.
     * High Openness + High Conscientiousness: Disciplined innovation and systematic creative execution.

4. CRISIS PROTOCOL & HALLUCINATION RESISTANCE:
   - If user context indicates acute self-harm, severe distress, or emergency, immediately withhold psychometric analysis and provide verified, official support resources (e.g., National Suicide Prevention / ER / local emergency services). Never hallucinate fictional hotlines.

5. TONE & EPISTEMIC MODESTY:
   - Maintain an objective, respectful, scientifically grounded, and descriptive tone.
   - Avoid pseudo-mystical Barnum statements ("you sometimes feel misunderstood"). Ground every observation directly in the provided facet percentiles.
"""


def build_analysis_prompt(profile: Dict[str, Any]) -> str:
    """
    Constructs the user-turn prompt encapsulating the standardized profile
    with instructions to return a rigorous JSON evaluation.
    """
    domains_summary = []
    for domain, data in profile.items():
        facets_str = ", ".join(
            [f"{f}: {v['percentile']}% ({v['band']})" for f, v in data.get("facets", {}).items()]
        )
        domains_summary.append(
            f"- {domain}: Percentile {data['percentile']}% (Band: {data['band']}, Raw: {data['raw_mean']})\n"
            f"  Facets: {facets_str}"
        )

    profile_text = "\n".join(domains_summary)

    return f"""Analyze the following standardized IPIP-NEO psychometric profile:

{profile_text}

Provide a structured, rigorous psychological synthesis formatted strictly as JSON with the following keys:
{{
  "executive_summary": "Concise 2-3 sentence overview of the core dimensional architecture.",
  "trait_interactions": [
    {{
      "interaction": "Name of the interacting domains (e.g. Neuroticism x Conscientiousness)",
      "observation": "Detailed behavioral mechanism resulting from this specific pairing."
    }}
  ],
  "epistemic_disclaimer": "Explicit note regarding state-vs-trait variance and non-diagnostic bounds.",
  "strengths": ["List of 3 concrete behavioral strengths grounded in the profile."],
  "development_areas": ["List of 2 constructive areas for self-regulation or environment optimization."]
}}
"""
