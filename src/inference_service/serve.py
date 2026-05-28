"""
FastAPI server implementing SageMaker's HTTP inference contract.
"""

import asyncio
import logging

from fastapi import FastAPI, Request, Response

from sagemaker_handler import model_load, parse_request, run_prediction, serialize_response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_bundle: dict = {}
_load_lock = asyncio.Lock()

async def _ensure_loaded():
    if "model" not in _bundle:
        async with _load_lock:
            if "model" not in _bundle:
                logger.info("Cold start: loading models...")
                loop = asyncio.get_event_loop()
                _bundle["model"] = await loop.run_in_executor(
                    None, model_load, "/opt/ml/model"
                )
                logger.info("Model bundle ready.")


app = FastAPI()

@app.get("/ping")
def health_check():
    """SageMaker calls this to confirm the container is healthy before routing traffic."""
    return Response(status_code=200)

# sagemaker requirement (not infer)
@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker routes every prediction request here."""
    await _ensure_loaded()

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    parsed = parse_request(body, content_type)
    result = run_prediction(parsed, _bundle["model"])
    response_body, media_type = serialize_response(result)

    return Response(content=response_body, media_type=media_type)
