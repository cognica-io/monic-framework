#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#


class SecurityError(Exception):
    """Raised when dangerous operations are detected."""


class UnsupportedUnpackingError(Exception):
    """Raised when an unsupported unpacking pattern is encountered."""
