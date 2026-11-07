"""
Unit tests for ofiLab Psychometric Scoring Engine.
"""

import math
import pytest
from scoring_engine import (
    FACET_TAXONOMY,
    EMPIRICAL_NORMS,
    _compute_normal_cdf,
    normalize_domain_score,
    build_synthetic_profile,
    DomainScore,
)


def test_facet_taxonomy_counts():
    assert len(FACET_TAXONOMY) == 5
    for domain, facets in FACET_TAXONOMY.items():
        assert len(facets) == 6, f"Domain {domain} must have exactly 6 facets"
    total_facets = sum(len(f) for f in FACET_TAXONOMY.values())
    assert total_facets == 30, "IPIP-NEO must have exactly 30 facets"


def test_normal_cdf_values():
    assert math.isclose(_compute_normal_cdf(0.0), 0.5, abs_tol=1e-5)
    assert _compute_normal_cdf(3.0) > 0.99
    assert _compute_normal_cdf(-3.0) < 0.01


def test_normalize_domain_score_bands():
    # Mean score should yield Average band (~50th percentile)
    norm_mean = EMPIRICAL_NORMS["Neuroticism"]["mean"]
    res = normalize_domain_score("Neuroticism", norm_mean)
    assert res.percentile == 50
    assert res.band == "Average"
    assert res.z_score == 0.0

    # Very high score
    res_high = normalize_domain_score("Neuroticism", 4.8)
    assert res_high.percentile > 70
    assert res_high.band == "High"

    # Very low score
    res_low = normalize_domain_score("Neuroticism", 1.2)
    assert res_low.percentile < 30
    assert res_low.band == "Low"


def test_build_synthetic_profile():
    profile = build_synthetic_profile()
    assert len(profile) == 5
    for domain in ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]:
        assert domain in profile
        d_data = profile[domain]
        assert 1 <= d_data["percentile"] <= 99
        assert d_data["band"] in ["Low", "Average", "High"]
        assert len(d_data["facets"]) == 6
        for f_name, f_val in d_data["facets"].items():
            assert 1.0 <= f_val["raw_mean"] <= 5.0
            assert 1 <= f_val["percentile"] <= 99
            assert f_val["band"] in ["Low", "Average", "High"]
