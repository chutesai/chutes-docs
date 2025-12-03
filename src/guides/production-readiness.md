# Production Readiness Guide

Moving from a prototype to a production-grade application on Chutes requires attention to reliability, security, and scaling. This checklist covers the essential steps to ensure your chute is ready for the real world.

## 1. Reliability & Stability

### ✅ Handle Startup & Shutdown
Ensure your `on_startup` logic is robust. Pre-download all necessary models and artifacts so the first request is fast.

```python
@chute.on_startup()
async def startup(self):
    # Fail fast if critical resources are missing
    if not os.path.exists("model.bin"):
        raise RuntimeError("Model file missing!")
    self.model = load_model("model.bin")
```

### ✅ Implement Health Checks
Define a lightweight cord for health monitoring (e.g., by load balancers).

```python
@chute.cord(public_api_path="/health", method="GET")
async def health(self):
    if self.model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ok"}
```

### ✅ Graceful Error Handling
Don't let internal errors crash your service or leak stack traces to users. Wrap logic in try/except blocks and return appropriate HTTP status codes.

```python
try:
    result = self.model.predict(data)
except ValueError:
    raise HTTPException(400, "Invalid input data")
except Exception as e:
    logger.error(f"Inference failed: {e}")
    raise HTTPException(500, "Internal inference error")
```

## 2. Performance & Scaling

### ✅ Concurrency Tuning
Set `concurrency` appropriately.
*   **1**: For heavy, atomic workloads (e.g., image generation) where batching isn't possible.
*   **High (e.g., 64+)**: For async engines like vLLM that handle internal batching.

### ✅ Auto-Scaling Configuration
Configure scaling parameters to handle traffic spikes without over-provisioning.

```python
chute = Chute(
    ...
    min_instances=1,           # Keep one warm if low latency is critical
    max_instances=10,          # Cap costs/resources
    scaling_threshold=0.75,    # Scale up when 75% utilized
    shutdown_after_seconds=300 # Scale down after 5 min idle
)
```

### ✅ Caching
Use internal caching (LRU) or external caches (Redis) for frequent, identical queries to save compute.

## 3. Security

### ✅ Scoped API Keys
Never use your admin API key in client-side code. Create scoped keys for specific functions.

```bash
# Create a key that can ONLY invoke this specific chute
chute keys create --name "app-client" --action invoke --chute-ids <your-chute-id>
```

### ✅ Input Validation
Use Pydantic schemas strictly. Validate string lengths, image sizes, and numeric ranges to prevent DOS attacks or memory overflows.

```python
class Input(BaseModel):
    prompt: str = Field(..., max_length=1000) # Prevent massive prompt attacks
    steps: int = Field(..., ge=1, le=50)      # Bound compute usage
```

## 4. Observability

### ✅ Logging
Log structured data (JSON) where possible for easy parsing. Log important events (startup, errors) but avoid logging sensitive user data (PII).

### ✅ Metrics
Use the built-in Prometheus client if you need custom metrics (e.g., "images_generated_total"), or rely on the platform's standard metrics (requests/sec, latency).

## 5. Deployment Strategy

### ✅ Pinned Versions
Always pin your dependencies in `requirements.txt` or your `Image` definition.
*   **Bad**: `pip install torch`
*   **Good**: `pip install torch==2.4.0`

### ✅ Immutable Tags
Don't rely on `latest` tags for base images. Use specific SHA digests or version tags to ensure reproducibility.

### ✅ Staging Environment
Deploy a separate "staging" chute (e.g., `my-app-staging`) to test changes before updating your production chute.

## Production Checklist Summary

- [ ] **Model Loading**: Pre-loaded on startup, not per-request.
- [ ] **Error Handling**: User-friendly HTTP errors, no stack traces.
- [ ] **Validation**: Strict Pydantic schemas for all inputs.
- [ ] **Scaling**: `max_instances` set to protect budget.
- [ ] **Security**: Scoped API keys generated for clients.
- [ ] **Dependencies**: All packages pinned to specific versions.
- [ ] **Monitoring**: Health check endpoint exists and works.

