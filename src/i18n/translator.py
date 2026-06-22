# -*- coding: utf-8 -*-
"""Translation service for the i18n system."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language


class Translator:
    """Enterprise-grade translation service with fallback and interpolation."""

    def __init__(self, translations_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the translator and load JSON files.

        Args:
            translations_dir: Directory containing language JSON files.
                Defaults to the 'translations' folder next to this file.
        """
        if translations_dir is None:
            self.translations_dir = Path(__file__).parent / "translations"
        else:
            self.translations_dir = Path(translations_dir)

        self._translations: Dict[str, Dict[str, str]] = {}
        self._load_all_languages()

    def _load_all_languages(self) -> None:
        """Load all JSON files in the translations directory."""
        if not self.translations_dir.exists():
            return

        for lang_enum in Language:
            file_path = self.translations_dir / f"{lang_enum.value}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._translations[lang_enum.value] = self._flatten_dict(data)
            else:
                self._translations[lang_enum.value] = {}

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, str]:
        """Flatten a nested dictionary into a single-level dict with dot notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def t(self, key: Union[TranslationKey, str], lang: Union[Language, str] = Language.EN, **kwargs: Any) -> str:
        """Translate a key using the specified language.

        Fallback strategy:
        1. Try to find the key in the requested language.
        2. Fall back to English ('en').
        3. Fall back to the key string itself.

        Args:
            key: The TranslationKey or string key (e.g. 'header.title').
            lang: The target language.
            **kwargs: Keyword arguments for string interpolation.

        Returns:
            The translated and interpolated string.
        """
        if isinstance(key, TranslationKey):
            key_str = key.value
        else:
            key_str = str(key)

        if isinstance(lang, Language):
            lang_code = lang.value
        else:
            lang_code = str(lang)

        # 1. Try requested language
        lang_dict = self._translations.get(lang_code, {})
        if key_str in lang_dict:
            template = lang_dict[key_str]
            return self._interpolate(template, key_str, **kwargs)

        # 2. Try English fallback
        en_dict = self._translations.get(Language.EN.value, {})
        if key_str in en_dict:
            template = en_dict[key_str]
            return self._interpolate(template, key_str, **kwargs)

        # 3. Fallback to key
        return self._interpolate(key_str, key_str, **kwargs)

    def _interpolate(self, template: str, key_str: str, **kwargs: Any) -> str:
        """Interpolate kwargs into the template string."""
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # If a keyword argument is missing, return the unformatted string
            # to avoid crashing the application.
            import logging
            logging.getLogger(__name__).warning(
                "Missing interpolation key %s for translation string '%s'", e, key_str
            )
            return template
