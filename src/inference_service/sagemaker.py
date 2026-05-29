"""
SageMaker endpoint handler for Ribbit AI frog species identification.
"""

import glob
import io
import json
import base64
import binascii
import logging
import os
import pickle

import boto3
import librosa
import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationError

_S3_BUCKET = "ml-stack-863631582176-us-west-2"
_S3_PREFIX = "Models/FSL_Species_71/"
_LOCAL_DIR = "/tmp/ribbit_models"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ClientError(ValueError):
    """Raised for invalid/malformed requests (maps to HTTP 4xx in SageMaker)."""

class ServerError(RuntimeError):
    """Raised for internal failures (maps to HTTP 5xx in SageMaker)."""

class AudioRequest(BaseModel):
    audio_b64: str
    latitude:  float | None = Field(default=None, ge=-90,  le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    user_id:   str   | None = Field(default=None, max_length=64)

    @field_validator("audio_b64")
    @classmethod
    def must_be_valid_base64(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio_b64 is not valid base64") from exc
        if len(decoded) == 0:
            raise ValueError("audio_b64 decodes to empty bytes — audio is required")
        if len(decoded) < 44:
            raise ValueError(
                f"audio_b64 decodes to only {len(decoded)} bytes — too short to be audio"
            )
        return v

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def model_load(_model_dir: str) -> dict:

    os.makedirs(_LOCAL_DIR, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1")

    logger.info("Downloading model artifacts from s3://%s/%s ...", _S3_BUCKET, _S3_PREFIX)
    for obj in s3.list_objects_v2(Bucket=_S3_BUCKET, Prefix=_S3_PREFIX).get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        dest = os.path.join(_LOCAL_DIR, os.path.basename(key))
        logger.info("  %s → %s", key, dest)
        s3.download_file(_S3_BUCKET, key, dest)

    with open(os.path.join(_LOCAL_DIR, "71_ensemble_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    round_files = sorted(glob.glob(os.path.join(_LOCAL_DIR, "model_71_round_*.pkl")))
#    logger.info("Loading %d ensemble models...", len(round_files))
    models = []
    for p in round_files:
        with open(p, "rb") as fh:
            models.append(pickle.load(fh))

#    logger.info("Loading BirdNET...")
    from birdnet_analyzer.model import load_model, embeddings as _birdnet_embed

    load_model(class_output=False)

#    logger.info("Model bundle ready.")
    return {
        "scaler": meta["scaler"],
        "num_classes": meta["num_classes"],
        "index_to_species": meta["index_to_species"],
        "models": models,
        "birdnet_embed": _birdnet_embed,
    }


def parse_request(body: bytes, content_type: str) -> dict:

    if content_type != "application/json":
        raise ClientError(
            f"Unsupported content type '{content_type}'. Send application/json."
        )

    try:
        raw = json.loads(body)

    except json.JSONDecodeError as exc:
        raise ClientError(f"Request body is not valid JSON: {exc}") from exc

    try:
        req = AudioRequest.model_validate(raw)

    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        logger.warning("Request validation failed: %s", messages)
        raise ClientError(f"Invalid request payload — {messages}") from exc

    return {
        "audio_bytes": base64.b64decode(req.audio_b64),
        "latitude":    req.latitude,
        "longitude":   req.longitude,
        "user_id":     req.user_id,
    }

def run_prediction(request: dict, bundle: dict) -> dict:
    try:
        audio, _ = librosa.load(io.BytesIO(request["audio_bytes"]), sr=48000, mono=True)
    except Exception as exc:
        logger.warning("Audio decoding failed: %s", exc)
        raise ClientError(
            "Could not decode audio — send a valid WAV or MP3 file."
        ) from exc

    if len(audio) == 0:
        raise ClientError("Audio file contains no samples.")

    target = 48_000 * 3
    audio = (
        np.pad(audio, (0, target - len(audio)), "constant")
        if len(audio) < target
        else audio[:target]
    )

    try:
        embedding = bundle["birdnet_embed"](np.expand_dims(audio, axis=0))
        scaled    = bundle["scaler"].transform(embedding)
        probs     = np.zeros((scaled.shape[0], bundle["num_classes"]))
        for model in bundle["models"]:
            probs += model.predict_proba(scaled)
        probs /= len(bundle["models"])
    except Exception as exc:
        logger.error("Inference pipeline error: %s", exc, exc_info=True)
        raise ServerError("Model inference failed — see container logs.") from exc

    top_idx = np.argsort(probs[0])[-5:][::-1]
    return {
        "predictions": [
            {"species": bundle["index_to_species"][i], "confidence": round(float(probs[0][i]), 4)}
            for i in top_idx
        ],
        "user_id":   request.get("user_id"),
        "latitude":  request.get("latitude"),
        "longitude": request.get("longitude"),
    }

def serialize_response(result: dict) -> tuple[str, str]:
    return json.dumps(result, cls=_NumpyEncoder), "application/json"


# ── SageMaker required hooks ───────────────────────────────────────────────────
_ERR = "__error__"


def model_fn(model_dir):
    return model_load(model_dir)


def input_fn(body, ct):
    try:
        return parse_request(body, ct)
    except ClientError as exc:
        return {_ERR: True, "error": "ClientError", "message": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error in input_fn: %s", exc, exc_info=True)
        return {_ERR: True, "error": "ServerError", "message": "Failed to parse request."}


def predict_fn(data, mdl):
    if isinstance(data, dict) and data.get(_ERR):
        return data
    try:
        return run_prediction(data, mdl)
    except ClientError as exc:
        return {_ERR: True, "error": "ClientError", "message": str(exc)}
    except ServerError as exc:
        return {_ERR: True, "error": "ServerError", "message": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error in predict_fn: %s", exc, exc_info=True)
        return {_ERR: True, "error": "ServerError", "message": "Inference failed unexpectedly."}


def output_fn(res, _accept):
    if isinstance(res, dict) and res.get(_ERR):
        body = json.dumps({"error": res["error"], "message": res["message"]})
        return body, "application/json"
    return serialize_response(res)