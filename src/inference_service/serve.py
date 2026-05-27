import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from sagemaker_handler import model_load, parse_request, run_prediction, serialize_response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cold start: loading models...")
    _bundle["model"] = model_load("/opt/ml/model")
    logger.info("Model bundle ready — server accepting requests.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def health_check():
    return Response(status_code=200)

# sagemaker requirement (not infer)
@app.post("/invocations")
async def invoke(request: Request):
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    parsed = parse_request(body, content_type)
    result = run_prediction(parsed, _bundle["model"])
    response_body, media_type = serialize_response(result)

    return Response(content=response_body, media_type=media_type)
