from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.dashboard import DASHBOARD_HTML
from app.predictor import ModelManager, ModelNotReady
from app.schemas import CustomerFeatures, Outcome, Prediction
from app.store import PredictionStore

manager = ModelManager()
store: PredictionStore | None = None
REQUESTS = Counter("ml_prediction_requests_total", "Prediction requests", ["decision"])
LATENCY = Histogram("ml_prediction_latency_seconds", "Prediction latency")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global store
    store = PredictionStore()
    try:
        manager.load(force=True)
    except ModelNotReady:
        pass
    yield


app = FastAPI(title="Production ML Decision Service", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    try:
        manager.load()
        return {"status": "ready", "model_version": manager.metadata["model_version"]}
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/model")
def model_info() -> dict:
    try:
        manager.load()
        return manager.metadata
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/predict", response_model=Prediction)
def predict(features: CustomerFeatures) -> dict:
    with LATENCY.time():
        try:
            payload = features.model_dump()
            result = manager.predict_one(payload)
        except ModelNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    REQUESTS.labels(result["decision"]).inc()
    if store is not None:
        store.save_prediction(result, payload)
    return result


@app.post("/v1/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[CustomerFeatures]) -> list[dict]:
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 1000")
    return [predict(customer) for customer in customers]


@app.post("/v1/outcomes", status_code=204)
def record_outcome(outcome: Outcome) -> Response:
    if store is None:
        raise HTTPException(status_code=503, detail="Outcome store unavailable")
    store.save_outcome(outcome.model_dump())
    return Response(status_code=204)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
