import math

from bench.llm_eval import parse_level
from bench.reference import hep, psf4_multiplier
from bench.spec import LEVELS


def test_parse_level_prefers_longest_match():
    lv = LEVELS[1]
    assert parse_level('{"level": "매우나쁨"}', lv) == "매우나쁨"
    assert parse_level("<think>보통일까</think>{\"level\": \"나쁨\"}", lv) == "나쁨"
    assert parse_level("답: 매우나쁨", lv) == "매우나쁨"
    assert parse_level("모르겠습니다", lv) is None


def test_psf4_formula_matches_railway_hep():
    assert math.isclose(psf4_multiplier(0), 0.33)
    assert math.isclose(psf4_multiplier(18.555), 0.33 + 4.5 * 0.5)


def test_hep_baseline_and_saturation():
    neutral = {1: "보통", 2: "보통", 3: "보통", 5: "보통", 6: "보통", 7: "보통", 8: "보통"}
    m4 = psf4_multiplier(12)
    assert math.isclose(hep("기관사", neutral, 12), (0.01225 + 0.00615) * m4)
    worst = {1: "매우나쁨", 2: "나쁨", 3: "나쁨", 5: "나쁨", 6: "나쁨", 7: "나쁨", 8: "매우 높음"}
    assert hep("기관사", worst, 48) < 2  # corrected formula keeps each part below 1


def test_fixed_form_sends_expert_direction():
    """Railway_hep-fixed must send 2 for the worse option of every item (direction confirmed by the survey)."""
    from pathlib import Path

    import pytest

    from bench.spec import ROOT, build_spec

    fixed = ROOT / "vendor" / "Railway_hep-fixed"
    if not (fixed / "input.html").exists():
        pytest.skip("fix worktree not checked out")
    original = build_spec()
    patched = build_spec(fixed)
    reversed_before = [o["code"] for s in original.values() for it in s["items"] for o in it["options"]
                       if o["value"] != o["value_web"]]
    assert reversed_before, "the original form should show the known reversal"
    for s in patched.values():
        for it in s["items"]:
            for o in it["options"]:
                assert o["value_web"] == o["value"], o["code"]
