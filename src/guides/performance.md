# Performance Optimization Guide

This comprehensive guide covers performance optimization strategies for Chutes applications, from model inference to network efficiency and resource management.

## Overview

Performance optimization in Chutes involves several key areas:

- **Model Optimization**: Quantization, compilation, and batching
- **Resource Management**: Efficient GPU and memory usage
- **Scaling Strategies**: Auto-scaling and load balancing
- **Caching**: Reducing redundant computations
- **Network Optimization**: Minimizing latency and payload size
- **Monitoring**: Tracking metrics to identify bottlenecks

## Model Inference Optimization

### Dynamic Batching

Processing requests in batches significantly improves GPU utilization. Here's a robust dynamic batcher implementation:

```python
import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class BatchRequest:
    data: Dict[str, Any]
    future: asyncio.Future
    timestamp: float

class DynamicBatcher:
    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.01):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: List[BatchRequest] = []
        self.processing = False
        self.lock = asyncio.Lock()

    async def add_request(self, data: Dict[str, Any]) -> Any:
        """Add request to batch queue"""
        future = asyncio.Future()
        request = BatchRequest(data, future, time.time())

        async with self.lock:
            self.pending_requests.append(request)
            if not self.processing:
                asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests"""
        async with self.lock:
            if self.processing or not self.pending_requests:
                return
            self.processing = True

        while True:
            # Wait for batch to accumulate or timeout
            start_time = time.time()
            while (len(self.pending_requests) < self.max_batch_size and
                   time.time() - start_time < self.max_wait_time):
                await asyncio.sleep(0.001)

            async with self.lock:
                if not self.pending_requests:
                    break
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]

            # Run inference
            try:
                batch_data = [req.data for req in batch]
                results = await self._run_inference(batch_data)
                
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

        async with self.lock:
            self.processing = False

    async def _run_inference(self, batch_data: List[Dict]) -> List[Any]:
        """Override this with your actual inference logic"""
        # Example:
        # inputs = tokenizer([item["text"] for item in batch_data], padding=True, return_tensors="pt")
        # outputs = model(**inputs)
        # return outputs
        return [{"result": "mock_result"} for _ in batch_data]
```

### Model Quantization

Reduce model size and memory footprint using quantization (e.g., 8-bit or 4-bit):

```python
from chutes.image import Image

# Build image with quantization support
image = (
    Image(username="myuser", name="quantized-model", tag="1.0")
    .pip_install([
        "torch",
        "transformers",
        "bitsandbytes",  # Required for 8-bit/4-bit
        "accelerate"
    ])
)

# Loading a quantized model
def load_quantized_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium",
        quantization_config=quant_config,
        device_map="auto"
    )
    return model
```

### TorchScript Compilation

Compile PyTorch models for faster execution:

```python
import torch

def optimize_model(model, example_input):
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    return torch.jit.freeze(traced_model)
```

## Resource Management

### GPU Memory Management

Properly managing GPU memory is critical to avoid OOM errors and maximize throughput.

```python
import torch
import gc
from contextlib import contextmanager

class GPUMemoryManager:
    @contextmanager
    def optimization_context(self):
        """Context manager to clear cache before and after operations"""
        self.cleanup()
        try:
            yield
        finally:
            self.cleanup()

    def cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_usage(self):
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

# Usage
memory_manager = GPUMemoryManager()

async def run_inference(inputs):
    with memory_manager.optimization_context():
        # Run heavy inference here
        pass
```

## Scaling Strategies

### Auto-scaling Configuration

Configure your chute to scale automatically based on load:

```python
from chutes.chute import Chute, NodeSelector

chute = Chute(
    # ... other args ...
    concurrency=10,  # Max concurrent requests per instance
    
    # Auto-scaling settings
    auto_scale=True,
    min_instances=1,
    max_instances=10,
    scale_up_threshold=0.8,    # Scale up when 80% concurrency reached
    scale_down_threshold=0.3,  # Scale down when <30% utilized
    scale_up_cooldown=60,      # Wait 60s before next scale up
    scale_down_cooldown=300    # Wait 5m before scaling down
)
```

## Caching Strategies

### Redis Caching

Use Redis for distributed caching across multiple chute instances:

```python
import redis
import pickle
import hashlib

class CacheManager:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    def get_key(self, prefix, *args, **kwargs):
        key_str = str(args) + str(sorted(kwargs.items()))
        return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"

    def get(self, key):
        data = self.redis.get(key)
        return pickle.loads(data) if data else None

    def set(self, key, value, ttl=3600):
        self.redis.setex(key, ttl, pickle.dumps(value))

# Decorator usage
def cached(ttl=3600):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            key = self.cache.get_key(func.__name__, *args, **kwargs)
            result = self.cache.get(key)
            if result:
                return result
            
            result = await func(self, *args, **kwargs)
            self.cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
```

## Network Optimization

### Response Compression

Compress large JSON responses to reduce network transfer time:

```python
import gzip
import json

def compress_response(data: dict) -> dict:
    json_str = json.dumps(data)
    if len(json_str) < 1024:  # Don't compress small responses
        return data
        
    compressed = gzip.compress(json_str.encode())
    return {
        "compressed": True,
        "data": compressed.hex()
    }
```

### Streaming

For long-running generations (like LLMs), use streaming to provide immediate feedback. See the [Streaming Guide](streaming) for details.

## Monitoring

Track performance metrics to identify bottlenecks.

```python
import time
from prometheus_client import Histogram, Counter

REQUEST_TIME = Histogram('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total request count')

@chute.cord(public_api_path="/run", method="POST")
async def run(self, data: dict):
    REQUEST_COUNT.inc()
    with REQUEST_TIME.time():
        # Process request
        return await self.process(data)
```

## Next Steps

- **[Cost Optimization](cost-optimization)**: Balance performance with cost
- **[Best Practices](best-practices)**: General deployment guidelines
- **[Streaming Guide](streaming)**: Implement real-time responses
