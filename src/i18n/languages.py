# -*- coding: utf-8 -*-
"""Language enumerations for the i18n system."""

from enum import Enum


class Language(str, Enum):
    """Supported languages."""
    IT = "it"
    EN = "en"
