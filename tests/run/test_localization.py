import json

import pytest

from minisweagent.run.benchmarks.utils.localization import (
    PATCH_DELIMITER,
    Localization,
    SubmissionParseError,
    split_submission,
)


_VALID_LOC = {
    "root_cause": [
        {
            "file": "astropy/utils/introspection.py",
            "line": 143,
            "statement": "return LooseVersion(have_version) >= LooseVersion(version)",
        }
    ],
    "reasoning": "LooseVersion comparison fails when version strings mix int and str components.",
    "confidence": "high",
}

_PATCH = (
    "diff --git a/foo.py b/foo.py\n"
    "index abc..def 100644\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1 +1 @@\n"
    "-bad\n"
    "+good\n"
)


def _submission(loc: dict, patch: str) -> str:
    return f"{json.dumps(loc)}\n{PATCH_DELIMITER}\n{patch}"


def test_split_submission_happy_path():
    localization, patch = split_submission(_submission(_VALID_LOC, _PATCH))
    assert isinstance(localization, Localization)
    assert localization.confidence == "high"
    assert len(localization.root_cause) == 1
    assert localization.root_cause[0].file == "astropy/utils/introspection.py"
    assert localization.root_cause[0].line == 143
    assert patch == _PATCH


def test_split_submission_multiple_root_causes():
    loc = {
        "root_cause": [
            {"file": "a.py", "line": 1, "statement": "x = 1"},
            {"file": "b.py", "line": 2, "statement": "y = 2"},
        ],
        "reasoning": "Both sites contribute to the bug.",
        "confidence": "medium",
    }
    localization, _ = split_submission(_submission(loc, _PATCH))
    assert len(localization.root_cause) == 2
    assert localization.root_cause[1].file == "b.py"


def test_split_submission_missing_delimiter():
    with pytest.raises(SubmissionParseError, match="Missing delimiter"):
        split_submission(json.dumps(_VALID_LOC) + "\n" + _PATCH)


def test_split_submission_empty_localization():
    with pytest.raises(SubmissionParseError, match="empty"):
        split_submission(f"\n{PATCH_DELIMITER}\n{_PATCH}")


def test_split_submission_invalid_json():
    with pytest.raises(SubmissionParseError, match="invalid"):
        split_submission(f"not json{{\n{PATCH_DELIMITER}\n{_PATCH}")


def test_split_submission_schema_violation_missing_field():
    loc = {"root_cause": [], "reasoning": "no causes"}  # missing confidence, empty root_cause
    with pytest.raises(SubmissionParseError, match="schema"):
        split_submission(_submission(loc, _PATCH))


def test_split_submission_schema_violation_bad_confidence():
    loc = dict(_VALID_LOC, confidence="very_high")
    with pytest.raises(SubmissionParseError, match="schema"):
        split_submission(_submission(loc, _PATCH))


def test_split_submission_schema_violation_wrong_types():
    loc = dict(_VALID_LOC, root_cause=[{"file": "a.py", "line": "not-an-int", "statement": "x"}])
    with pytest.raises(SubmissionParseError, match="schema"):
        split_submission(_submission(loc, _PATCH))


def test_split_submission_preserves_patch_with_delimiter_like_text_only_once():
    # Only the first occurrence of the delimiter is used as a separator.
    patch_with_text = _PATCH + "\n+ harmless line\n"
    localization, patch = split_submission(_submission(_VALID_LOC, patch_with_text))
    assert localization.confidence == "high"
    assert patch == patch_with_text
