"""
Unit tests for sagemaker.py hook error handling
Tests only input_fn / predict_fn / output_fn
Run with:
    python -m pytest ribbit_ai/src/inference_service/test_sagemaker.py -v
"""

import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sagemaker import _ERR, input_fn, output_fn, predict_fn

_FAKE_AUDIO_B64 = base64.b64encode(b"\x00" * 100).decode()

def test_wrong_content_type_returns_client_error():
    result = input_fn(b'{"audio_b64": "abc"}', "text/plain")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "content type" in result["message"].lower()


def test_invalid_json_returns_client_error():
    result = input_fn(b"not json at all {{{{", "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "json" in result["message"].lower()


def test_missing_audio_b64_returns_client_error():
    payload = json.dumps({"latitude": -23.5}).encode()
    result = input_fn(payload, "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "audio_b64" in result["message"]


def test_invalid_base64_returns_client_error():
    payload = json.dumps({"audio_b64": "!!!not-base64!!!"}).encode()
    result = input_fn(payload, "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"


def test_latitude_out_of_range_returns_client_error():
    payload = json.dumps({"audio_b64": _FAKE_AUDIO_B64, "latitude": 999.0}).encode()
    result = input_fn(payload, "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "latitude" in result["message"]


def test_longitude_out_of_range_returns_client_error():
    payload = json.dumps({"audio_b64": _FAKE_AUDIO_B64, "longitude": -999.0}).encode()
    result = input_fn(payload, "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "longitude" in result["message"]


def test_user_id_too_long_returns_client_error():
    payload = json.dumps({"audio_b64": _FAKE_AUDIO_B64, "user_id": "x" * 65}).encode()
    result = input_fn(payload, "application/json")
    assert result[_ERR] is True
    assert result["error"] == "ClientError"


def test_valid_payload_passes_input_fn():
    payload = json.dumps({
        "audio_b64": _FAKE_AUDIO_B64,
        "latitude": -23.5,
        "longitude": -46.6,
        "user_id": "alice",
    }).encode()
    result = input_fn(payload, "application/json")
    assert _ERR not in result
    assert "audio_bytes" in result


# ── predict_fn ─────────────────────────────────────────────────────────────────
def test_predict_fn_passes_error_envelope_through():
    envelope = {_ERR: True, "error": "ClientError", "message": "bad audio"}
    result = predict_fn(envelope, mdl={})
    assert result is envelope


def test_predict_fn_rejects_corrupt_audio():
    payload = json.dumps({"audio_b64": _FAKE_AUDIO_B64}).encode()
    parsed = input_fn(payload, "application/json")
    assert _ERR not in parsed, "should have passed input_fn"

    result = predict_fn(parsed, mdl={"birdnet_embed": None, "scaler": None,
                                     "num_classes": 5, "models": [],
                                     "index_to_species": {}})
    assert result[_ERR] is True
    assert result["error"] == "ClientError"
    assert "audio" in result["message"].lower()


# ── output_fn ──────────────────────────────────────────────────────────────────
def test_output_fn_formats_error_as_json():
    envelope = {_ERR: True, "error": "ClientError", "message": "bad payload"}
    body, content_type = output_fn(envelope, "application/json")
    assert content_type == "application/json"
    parsed = json.loads(body)
    assert parsed["error"] == "ClientError"
    assert parsed["message"] == "bad payload"
    assert _ERR not in parsed

def test_output_fn_formats_success_as_json():
    result = {
        "predictions": [{"species": "BOAFAB", "confidence": 0.87}],
        "user_id": "alice",
        "latitude": -23.5,
        "longitude": -46.6,
    }
    body, content_type = output_fn(result, "application/json")
    assert content_type == "application/json"
    parsed = json.loads(body)
    assert parsed["predictions"][0]["species"] == "BOAFAB"
