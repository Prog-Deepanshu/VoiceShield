"""
Unit tests for audio validation utilities.
"""

import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO

from app.utils.validation import (
    sanitize_filename,
    validate_audio_duration,
    validate_file_content,
    validate_file_meta,
)


def test_sanitize_filename():
    assert sanitize_filename("test.wav") == "test.wav"
    assert sanitize_filename("../../etc/passwd.wav") == "passwd.wav"
    assert sanitize_filename("C:\\Windows\\system32\\audio.mp3") == "audio.mp3"
    assert sanitize_filename("bad:name?.flac") == "badname.flac"


def test_validate_file_meta_valid():
    upload = UploadFile(filename="speech.wav", file=BytesIO(b"fake"), headers={"content-type": "audio/wav"})
    name = validate_file_meta(upload)
    assert name == "speech.wav"


def test_validate_file_meta_unsupported_ext():
    upload = UploadFile(filename="malicious.exe", file=BytesIO(b"fake"))
    with pytest.raises(HTTPException) as exc:
        validate_file_meta(upload)
    assert exc.value.status_code == 415


def test_validate_file_content_empty():
    with pytest.raises(HTTPException) as exc:
        validate_file_content(b"")
    assert exc.value.status_code == 400


def test_validate_file_content_size_limit():
    huge_data = b"0" * (51 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc:
        validate_file_content(huge_data, max_mb=50)
    assert exc.value.status_code == 413


def test_validate_audio_duration():
    # Valid
    validate_audio_duration(5.0)

    # Too short
    with pytest.raises(HTTPException) as exc:
        validate_audio_duration(0.2)
    assert exc.value.status_code == 400

    # Too long
    with pytest.raises(HTTPException) as exc:
        validate_audio_duration(300.0)
    assert exc.value.status_code == 400
