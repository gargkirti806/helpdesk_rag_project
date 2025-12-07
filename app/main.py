from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from .router import router

# -----------------------------
# Prometheus metrics setup
# -----------------------------
REQUEST_COUNT = Counter(
    'helpdesk_requests_total', 
    'Total number of helpdesk requests', 
    ['endpoint']
)
REQUEST_LATENCY = Histogram(
    'helpdesk_request_latency_seconds', 
    'Latency of helpdesk requests in seconds', 
    ['endpoint']
)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Helpdesk LangGraph RAG API",
    description="RAG-based helpdesk API with LangGraph workflow",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Pre-load models and initialize workflow at startup"""
    print("🚀 Starting up: Pre-loading models...")
    
    # Import and initialize workflow here to load models at startup
    from app.pipeline.graph import workflow
    
    # Optionally run a dummy invocation to ensure everything is loaded
    # This will trigger model loading once at startup
    try:
        dummy_config = {
            "configurable": {
                "thread_id": "startup_warmup",
                "checkpoint_ns": "warmup",
            }
        }
        # Run a minimal test to warm up the model
        print("🔄 Warming up the model...")
        # result = workflow.invoke({"user_query": "hello"}, config=dummy_config)
        # print(f"✅ Model loaded and ready! Warmup result: {type(result)}")
    except Exception as e:
        print(f"⚠️  Warning during warmup: {str(e)}")
        print("Server will continue, but first request may be slow.")
        import traceback
        traceback.print_exc()

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Helpdesk RAG server is running!"}

# Middleware to capture metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    
    # Skip metrics for static docs endpoints
    if endpoint in ["/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)
    
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)
    return response

# Endpoint to expose metrics
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)