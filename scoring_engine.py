"""
ofiLab Core - Psychometric Scoring Engine
Reference Implementation for IPIP-NEO 30-Facet Normalization

Grounding:
- International Personality Item Pool (Goldberg, 1999)
- 300/120-item IPIP-NEO representation (Johnson, 2014)
- Empirical normative standardizations (mean=0, sd=1 -> percentile transform)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 30-Facet Taxonomy according to IPIP-NEO
FACET_TAXONOMY: Dict[str, List[str]] = {
    "Neuroticism": [
        "Anxiety",
        "Anger",
        "Depression",
        "Self-Consciousness",
        "Immoderation",
        "Vulnerability",
    ],
    "Extraversion": [
        "Friendliness",
        "Gregariousness",
        "Assertiveness",
        "Activity Level",
        "Excitement-Seeking",
        "Cheerfulness",
    ],
    "Openness": [
        "Imagination",
        "Artistic Interests",
        "Emotionality",
        "Adventurousness",
        "Intellect",
        "Liberalism",
    ],
    "Agreeableness": [
        "Trust",
        "Morality",
        "Altruism",
        "Cooperation",
        "Modesty",
        "Sympathy",
    ],
    "Conscientiousness": [
        "Self-Efficacy",
        "Orderliness",
        "Dutifulness",
        "Achievement-Striving",
        "Self-Discipline",
        "Cautiousness",
    ],
}

# Empirical Normative Means and Standard Deviations (Johnson dataset, standardized scale 1-5)
EMPIRICAL_NORMS: Dict[str, Dict[str, float]] = {
    "Neuroticism": {"mean": 2.78, "sd": 0.68},
    "Extraversion": {"mean": 3.24, "sd": 0.62},
    "Openness": {"mean": 3.61, "sd": 0.54},
    "Agreeableness": {"mean": 3.75, "sd": 0.51},
    "Conscientiousness": {"mean": 3.48, "sd": 0.58},
}


@dataclass
class FacetScore:
    facet_name: str
    raw_mean: float
    percentile: int
    band: str  # "Low", "Average", "High"


@dataclass
class DomainScore:
    domain_name: str
    raw_mean: float
    z_score: float
    percentile: int
    band: str
    facets: Dict[str, FacetScore] = field(default_factory=dict)


def _compute_normal_cdf(z: float) -> float:
    """Computes the cumulative distribution function for standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normalize_domain_score(domain_name: str, raw_mean: float) -> DomainScore:
    """
    Transforms raw domain mean (1.0 to 5.0 scale) into standardized z-score
    and empirical percentile based on peer-reviewed normative distributions.
    """
    norms = EMPIRICAL_NORMS.get(domain_name, {"mean": 3.0, "sd": 0.6})
    z = (raw_mean - norms["mean"]) / norms["sd"]
    percentile = int(round(_compute_normal_cdf(z) * 100))
    percentile = max(1, min(99, percentile))

    if percentile < 30:
        band = "Low"
    elif percentile <= 70:
        band = "Average"
    else:
        band = "High"

    return DomainScore(
        domain_name=domain_name,
        raw_mean=round(raw_mean, 2),
        z_score=round(z, 2),
        percentile=percentile,
        band=band,
    )


def build_synthetic_profile(
    domain_ratings: Optional[Dict[str, float]] = None
) -> Dict[str, Dict]:
    """
    Constructs a standardized profile payload suitable for LLM reasoning inputs.
    Defaults to a balanced baseline profile if no ratings are provided.
    """
    if domain_ratings is None:
        domain_ratings = {
            "Neuroticism": 3.10,
            "Extraversion": 3.40,
            "Openness": 3.85,
            "Agreeableness": 3.65,
            "Conscientiousness": 3.90,
        }

    profile: Dict[str, Dict] = {}
    for domain, raw in domain_ratings.items():
        norm = normalize_domain_score(domain, raw)
        domain_norms = EMPIRICAL_NORMS.get(domain, {"mean": 3.0, "sd": 0.6})
        facets_dict: Dict[str, Dict] = {}
        # Populate facet placeholders with slight variance around domain mean
        for i, facet in enumerate(FACET_TAXONOMY.get(domain, [])):
            facet_mean = max(1.0, min(5.0, raw + ((i % 3) - 1) * 0.15))
            f_z = (facet_mean - domain_norms["mean"]) / domain_norms["sd"]
            f_pct = max(1, min(99, int(round(_compute_normal_cdf(f_z) * 100))))
            f_band = "Low" if f_pct < 30 else ("Average" if f_pct <= 70 else "High")
            facets_dict[facet] = {
                "raw_mean": round(facet_mean, 2),
                "percentile": f_pct,
                "band": f_band,
            }

        profile[domain] = {
            "raw_mean": norm.raw_mean,
            "z_score": norm.z_score,
            "percentile": norm.percentile,
            "band": norm.band,
            "facets": facets_dict,
        }

    return profile


if __name__ == "__main__":
    import json
    sample = build_synthetic_profile()
    print(json.dumps(sample, indent=2))
