"""Parse bug-localization metadata submitted alongside the patch."""

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

PATCH_DELIMITER = "<<<MSWEA_PATCH_DELIMITER>>>"


class RootCause(BaseModel):
    file: str
    line: int
    statement: str


class Localization(BaseModel):
    root_cause: list[RootCause]
    reasoning: str
    confidence: Literal["high", "medium", "low"]


class SubmissionParseError(ValueError):
    """Raised when the submission does not contain a valid localization + patch pair."""


def split_submission(submission: str) -> tuple[Localization, str]:
    """Split an agent submission into (localization, patch).

    Expected layout (after the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker has
    already been stripped by the environment):

        <localization.json contents>
        <<<MSWEA_PATCH_DELIMITER>>>
        <unified diff>

    Raises SubmissionParseError if the delimiter is missing, the JSON is
    malformed, or the JSON does not match the Localization schema.
    """
    if PATCH_DELIMITER not in submission:
        raise SubmissionParseError(
            f"Missing delimiter '{PATCH_DELIMITER}' — submission must contain a "
            "localization JSON object before the patch."
        )
    loc_text, _, patch = submission.partition(PATCH_DELIMITER)
    loc_text = loc_text.strip()
    if not loc_text:
        raise SubmissionParseError("Localization section is empty.")
    try:
        loc_data = json.loads(loc_text)
    except json.JSONDecodeError as e:
        raise SubmissionParseError(f"Localization JSON is invalid: {e}") from e
    try:
        localization = Localization.model_validate(loc_data)
    except ValidationError as e:
        raise SubmissionParseError(f"Localization does not match required schema: {e}") from e
    return localization, patch.lstrip("\n")
