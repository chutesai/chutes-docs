# Best Practices for Production-Ready Chutes

This comprehensive guide covers production best practices for building, deploying, and maintaining robust, scalable, and secure Chutes applications in production environments.

## Overview

Production-ready Chutes applications require:

- **Scalable Architecture**: Design for growth and varying loads
- **Security**: Protect data, models, and infrastructure
- **Performance**: Optimize for speed, memory, and resource efficiency
- **Reliability**: Handle failures gracefully with high availability
- **Monitoring**: Complete observability and alerting
- **Maintainability**: Code quality, documentation, and operational procedures

## Application Architecture

### Modular Design Patterns

```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, Any, Optional, Dict
from dataclasses import dataclass
import logging

# Define clear interfaces
class ModelInterface(Protocol):
    """Protocol for AI model implementations."""

    async def load(self) -> None:
        """Load the model into memory."""
        ...

    async def predict(self, input_data: Any) -> Any:
        """Make prediction on input data."""
        ...

    async def unload(self) -> None:
        """Unload model from memory."""
        ...

class CacheInterface(Protocol):
    """Protocol for caching implementations."""

    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...

# Implement dependency injection
@dataclass
class Dependencies:
    """Application dependencies container."""

    model: ModelInterface
    cache: CacheInterface
    logger: logging.Logger
    metrics: Any  # Metrics collector
    config: Dict[str, Any]

class ServiceBase(ABC):
    """Base class for application services."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.logger = deps.logger
        self.model = deps.model
        self.cache = deps.cache

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass

class TextGenerationService(ServiceBase):
    """Text generation service implementation."""

    async def initialize(self) -> None:
        """Initialize text generation service."""
        await self.model.load()
        self.logger.info("Text generation service initialized")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with caching and error handling."""

        # Create cache key
        cache_key = self._create_cache_key(prompt, kwargs)

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            self.logger.info("Cache hit for text generation")
            return cached_result

        # Generate new result
        try:
            result = await self.model.predict(prompt, **kwargs)

            # Cache result
            await self.cache.set(cache_key, result, ttl=3600)

            return result

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    def _create_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Create deterministic cache key."""
        import hashlib
        import json

        key_data = {"prompt": prompt, "params": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True)
        return f"text_gen:{hashlib.md5(key_str.encode()).hexdigest()}"

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.model.unload()
        self.logger.info("Text generation service cleaned up")

# Chute implementation with dependency injection
from chutes.chute import Chute
chute = Chute(username="production", name="text-service")

@chute.on_startup()
async def initialize_app(self):
    """Initialize application with proper dependency injection."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("text-service")

    # Initialize model
    model = await self._create_model()

    # Initialize cache
    cache = await self._create_cache()

    # Initialize metrics
    metrics = await self._create_metrics()

    # Load configuration
    config = await self._load_config()

    # Create dependencies container
    self.deps = Dependencies(
        model=model,
        cache=cache,
        logger=logger,
        metrics=metrics,
        config=config
    )

    # Initialize services
    self.text_service = TextGenerationService(self.deps)
    await self.text_service.initialize()

async def _create_model(self):
    """Factory method for model creation."""
    # Implementation depends on your specific model
    pass

async def _create_cache(self):
    """Factory method for cache creation."""
    # Could be Redis, Memcached, or in-memory cache
    pass
async def _create_metrics(self):
    pass
async def _load_config(self):
    return {}
```

### Configuration Management

```python
import os
from typing import Optional, Union
from pydantic import BaseSettings, Field, validator
from pathlib import Path

class ApplicationConfig(BaseSettings):
    """Production application configuration."""

    # Environment
    environment: str = Field("production", env="APP_ENV")
    debug: bool = Field(False, env="APP_DEBUG")

    # Model settings
    model_name: str = Field(..., env="MODEL_NAME")
    model_path: Optional[str] = Field(None, env="MODEL_PATH")
    max_batch_size: int = Field(8, env="MAX_BATCH_SIZE")

    # Performance settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    request_timeout: float = Field(30.0, env="REQUEST_TIMEOUT")
    max_memory_usage: float = Field(0.9, env="MAX_MEMORY_USAGE")

    # Cache settings
    cache_backend: str = Field("redis", env="CACHE_BACKEND")
    cache_url: str = Field("redis://localhost:6379", env="CACHE_URL")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")

    # Security
    api_key_required: bool = Field(True, env="API_KEY_REQUIRED")
    allowed_origins: list = Field(["*"], env="ALLOWED_ORIGINS")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")

    # Monitoring
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    @validator('max_memory_usage')
    def validate_memory_usage(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('Memory usage must be between 0.1 and 1.0')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Environment-specific configurations
class DevelopmentConfig(ApplicationConfig):
    """Development environment configuration."""

    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    api_key_required: bool = False

class StagingConfig(ApplicationConfig):
    """Staging environment configuration."""

    environment: str = "staging"
    debug: bool = False
    log_level: str = "INFO"

class ProductionConfig(ApplicationConfig):
    """Production environment configuration."""

    environment: str = "production"
    debug: bool = False
    log_level: str = "WARNING"
    api_key_required: bool = True

def get_config() -> ApplicationConfig:
    """Get configuration based on environment."""

    env = os.getenv("APP_ENV", "production").lower()

    config_classes = {
        "development": DevelopmentConfig,
        "staging": StagingConfig,
        "production": ProductionConfig
    }

    config_class = config_classes.get(env, ProductionConfig)
    # Note: In a real app you'd instantiate this properly with env vars
    # return config_class() 
    return config_class(model_name="default-model") # simplified for example

# Usage in Chute
@chute.on_startup()
async def load_configuration(self):
    """Load and validate configuration."""
    self.config = get_config()

    # Configure logging based on config
    import logging
    logging.basicConfig(
        level=getattr(logging, self.config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    self.logger = logging.getLogger(f"chute.{self.config.environment}")
    self.logger.info(f"Application started in {self.config.environment} mode")
```

## Performance Optimization

See the [Performance Optimization Guide](performance) for detailed strategies. Key areas include:

- **Dynamic Batching**: Group requests for efficient GPU usage.
- **Caching**: Cache expensive model outputs using Redis or in-memory stores.
- **Quantization**: Use 8-bit or 4-bit quantization to reduce memory footprint and increase speed.
- **Async Processing**: Use async/await to handle concurrent requests without blocking.

## Security Best Practices

See the [Security Guide](security) for a deep dive. Essentials:

- **Authentication**: Always use API keys or JWTs in production.
- **Input Validation**: Validate and sanitize all inputs using Pydantic schemas.
- **Rate Limiting**: Prevent abuse by limiting requests per user/IP.
- **Secrets Management**: Use environment variables or mounted volumes for secrets; never hardcode them.

## Monitoring and Observability

Implement structured logging and metrics to track the health of your application.

```python
import time
from contextlib import contextmanager
from datetime import datetime
import json
import logging

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        # Configure JSON handler...

    def info(self, message, **kwargs):
        self.logger.info(json.dumps({"message": message, **kwargs}))

class PerformanceMonitor:
    def __init__(self):
        # Initialize prometheus metrics...
        pass

    @contextmanager
    def measure_request(self, endpoint):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            # Record metric...
```

## Deployment Best Practices

### Production Deployment Checklist

```python
class ProductionDeploymentChecklist:
    """Comprehensive production deployment checklist."""

    CHECKLIST = {
        "Security": [
            "✓ Enable HTTPS/TLS encryption",
            "✓ Configure API authentication",
            "✓ Set up rate limiting",
            "✓ Sanitize all inputs",
            "✓ Secrets management",
        ],

        "Performance": [
            "✓ Load testing completed",
            "✓ Memory usage optimized",
            "✓ Caching implemented",
            "✓ Auto-scaling rules configured",
        ],

        "Reliability": [
            "✓ Health checks implemented",
            "✓ Error handling comprehensive",
            "✓ Graceful shutdown handled",
        ],

        "Monitoring": [
            "✓ Application metrics",
            "✓ Error tracking",
            "✓ Log aggregation",
            "✓ Alert configuration",
        ],
        }
```

## Summary and Next Steps

This guide covers the essential patterns for building production-grade Chutes.

### Implementation Priority

1.  **Security**: Authentication and input validation.
2.  **Monitoring**: Logging and basic metrics.
3.  **Performance**: Caching and resource management.
4.  **Reliability**: Error handling and health checks.

For more specific guides, see:

- [Error Handling Guide](error-handling)
- [Custom Images Guide](custom-images)
- [Streaming Guide](streaming)
- [Templates Guide](templates)
- [Performance Optimization](performance)
