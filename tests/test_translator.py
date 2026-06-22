# -*- coding: utf-8 -*-
"""Unit tests for the Translator service."""

import json
from pathlib import Path

import pytest

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator


@pytest.fixture
def temp_translations_dir(tmp_path: Path) -> Path:
    """Create a temporary translations directory with mock JSON files."""
    translations_dir = tmp_path / "translations"
    translations_dir.mkdir()

    it_data = {
        "header": {
            "title": "Titolo IT"
        },
        "info": {
            "success": "Completato in {time}s"
        }
    }
    
    en_data = {
        "header": {
            "title": "Title EN",
            "subtitle": "Subtitle EN"
        },
        "info": {
            "success": "Completed in {time}s"
        }
    }

    with open(translations_dir / "it.json", "w", encoding="utf-8") as f:
        json.dump(it_data, f)
        
    with open(translations_dir / "en.json", "w", encoding="utf-8") as f:
        json.dump(en_data, f)

    return translations_dir


@pytest.fixture
def translator(temp_translations_dir: Path) -> Translator:
    return Translator(translations_dir=temp_translations_dir)


def test_flatten_dict() -> None:
    t = Translator(translations_dir=Path("dummy"))
    nested = {
        "a": {
            "b": {
                "c": "val1"
            },
            "d": "val2"
        },
        "e": "val3"
    }
    flat = t._flatten_dict(nested)
    assert flat == {
        "a.b.c": "val1",
        "a.d": "val2",
        "e": "val3"
    }


def test_translate_existing_key(translator: Translator) -> None:
    assert translator.t("header.title", lang=Language.IT) == "Titolo IT"
    assert translator.t("header.title", lang=Language.EN) == "Title EN"


def test_fallback_to_english(translator: Translator) -> None:
    # "header.subtitle" exists only in EN
    assert translator.t("header.subtitle", lang=Language.IT) == "Subtitle EN"


def test_fallback_to_key(translator: Translator) -> None:
    # "non.existent" exists nowhere
    assert translator.t("non.existent", lang=Language.IT) == "non.existent"


def test_string_interpolation(translator: Translator) -> None:
    assert translator.t("info.success", lang=Language.IT, time=1.5) == "Completato in 1.5s"
    assert translator.t("info.success", lang=Language.EN, time=2.0) == "Completed in 2.0s"


def test_missing_interpolation_kwargs(translator: Translator) -> None:
    # Missing kwargs should return the unformatted template safely
    res = translator.t("info.success", lang=Language.IT)
    assert res == "Completato in {time}s"


def test_translation_key_enum(translator: Translator) -> None:
    # We can mock a TranslationKey for testing
    class MockKey(str):
        @property
        def value(self):
            return "header.title"
            
    assert translator.t(MockKey(), lang=Language.IT) == "Titolo IT"
