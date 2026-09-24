"""Classify inference failures that may be retried without changing the score."""

import re


_RETRYABLE_API_ERROR = re.compile(
    r"Error during API call|API Error|Too many API failures|Request timed out|"
    r"Error code: (?:429|5\d\d)\b|502 Bad Gateway|engine is currently overloaded",
    re.IGNORECASE,
)


def is_retryable_api_failure(model_response: object) -> bool:
    return bool(_RETRYABLE_API_ERROR.search(str(model_response or "")))
