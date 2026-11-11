"""Input validation errors with user-readable messages."""

from __future__ import annotations


class ValidationError(ValueError):
    """Raised when inputs are physically impossible or logically inconsistent.

    `errors` holds one plain-English message per problem so the UI can list them all.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("Invalid inputs:\n- " + "\n- ".join(self.errors))
