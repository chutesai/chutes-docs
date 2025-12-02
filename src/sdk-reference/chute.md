# Chute API Reference

The `Chute` class is the core component of the Chutes framework, representing a deployable AI application unit. It extends FastAPI, so you can use all FastAPI features. This reference covers all methods, properties, and configuration options.

## Class Definition

```python
from chutes.chute import Chute

chute = Chute(
    username: str,
    name: str,
    image: str | Image,
    tagline: str = "",
    readme: str = "",
    standard_template: str = None,
    revision: str = None,
    node_selector: NodeSelector = None,
    concurrency: int = 1,
    max_instances: int = 1,
    shutdown_after_seconds: int = 300,
    scaling_threshold: float = 0.75,
    allow_external_egress: bool = False,
    encrypted_fs: bool = False,
    passthrough_headers: dict = {},
    tee: bool = False,
    **kwargs
)
```

## Constructor Parameters

### Required Parameters

#### `username: str`

The username or organization name for the chute deployment.

**Example:**

```python
chute = Chute(username="mycompany", name="ai-service", image="parachutes/python:3.12")
```

#### `name: str`

The name of the chute application.

**Example:**

```python
chute = Chute(username="mycompany", name="text-generator", image="parachutes/python:3.12")
```

#### `image: str | Image`

Docker image for the chute runtime environment (required).

**Example:**

```python
# Using a string reference to a pre-built image
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12"
)

# Using a custom Image object
from chutes.image import Image
custom_image = Image(username="mycompany", name="custom-ai", tag="1.0")
chute = Chute(
    username="mycompany",
    name="text-generator",
    image=custom_image
)
```

### Optional Parameters

#### `tagline: str = ""`

A brief description of what the chute does.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    tagline="Advanced text generation with GPT models"
)
```

#### `readme: str = ""`

Detailed documentation for the chute in Markdown format.

**Example:**

```python
readme = """
# Text Generation API

This chute provides advanced text generation capabilities.

## Features
- Multiple model support
- Customizable parameters
- Real-time streaming
"""

chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    readme=readme
)
```

#### `standard_template: str = None`

Reference to a standard template (e.g., "vllm", "sglang", "diffusion").

#### `revision: str = None`

Specific revision or version identifier for the chute.

#### `node_selector: NodeSelector = None`

Hardware requirements and preferences for the chute.

**Example:**

```python
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=24,
    include=["h100", "a100"],
    exclude=["t4"]
)

chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    node_selector=node_selector
)
```

#### `concurrency: int = 1`

Maximum number of concurrent requests the chute can handle per instance.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    concurrency=8  # Handle up to 8 concurrent requests
)
```

**Guidelines:**

- For vLLM/SGLang with continuous batching: 64-256
- For single-request models (diffusion): 1
- For models with some parallelism: 4-16

#### `max_instances: int = 1`

Maximum number of instances that can be scaled up.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    max_instances=10  # Scale up to 10 instances
)
```

#### `shutdown_after_seconds: int = 300`

Time in seconds to wait before shutting down an idle instance. Default is 5 minutes.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="parachutes/python:3.12",
    shutdown_after_seconds=600  # Shutdown after 10 minutes idle
)
```

#### `scaling_threshold: float = 0.75`

Utilization threshold at which to trigger scaling (0.0 to 1.0).

#### `allow_external_egress: bool = False`

Whether to allow external network connections after startup.

**Important:** By default, external network access is blocked after initialization. Set to `True` if your chute needs to fetch external resources at runtime (e.g., image URLs for vision models).

**Example:**

```python
# For vision language models that need to fetch images
chute = Chute(
    username="mycompany",
    name="vision-model",
    image="parachutes/python:3.12",
    allow_external_egress=True
)
```

#### `encrypted_fs: bool = False`

Whether to use encrypted filesystem for the chute.

#### `passthrough_headers: dict = {}`

Headers to pass through to passthrough cord endpoints.

#### `tee: bool = False`

Whether this chute runs in a Trusted Execution Environment.

#### `**kwargs`

Additional keyword arguments passed to the underlying FastAPI application.

## Decorators

### Lifecycle Decorators

#### `@chute.on_startup(priority: int = 50)`

Decorator for functions to run during chute startup.

**Signature:**

```python
@chute.on_startup(priority: int = 50)
async def initialization_function(self) -> None:
    """Function to run on startup."""
    pass
```

**Parameters:**

- `priority`: Execution order (lower values execute first, default=50)
  - 0-20: Early initialization
  - 30-70: Normal operations
  - 80-100: Late initialization

**Example:**

```python
@chute.on_startup(priority=10)  # Runs early
async def load_model(self):
    """Load the AI model during startup."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    self.model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("Model loaded successfully")

@chute.on_startup(priority=90)  # Runs late
async def log_startup(self):
    print("All initialization complete")
```

**Use Cases:**

- Load AI models
- Initialize databases
- Set up caches
- Configure services

#### `@chute.on_shutdown(priority: int = 50)`

Decorator for functions to run during chute shutdown.

**Signature:**

```python
@chute.on_shutdown(priority: int = 50)
async def cleanup_function(self) -> None:
    """Function to run on shutdown."""
    pass
```

**Example:**

```python
@chute.on_shutdown(priority=10)
async def cleanup_resources(self):
    """Clean up resources during shutdown."""
    if hasattr(self, 'model'):
        del self.model
    print("Resources cleaned up")
```

### API Endpoint Decorator

#### `@chute.cord()`

Decorator to create HTTP API endpoints. See [Cord Decorator Reference](/docs/sdk-reference/cord) for detailed documentation.

**Basic Example:**

```python
@chute.cord(public_api_path="/generate", public_api_method="POST")
async def generate_text(self, prompt: str) -> str:
    """Generate text from a prompt."""
    return await self.model.generate(prompt)
```

### Job Decorator

#### `@chute.job()`

Decorator to create long-running jobs or server rentals. See [Job Decorator Reference](/docs/sdk-reference/job) for detailed documentation.

**Basic Example:**

```python
from chutes.chute.job import Port

@chute.job(ports=[Port(name="web", port=8080, proto="http")], timeout=3600)
async def training_job(self, **job_data):
    """Long-running training job."""
    output_dir = job_data["output_dir"]
    # Perform training...
    return {"status": "completed"}
```

## Properties

### `chute.name`

The name of the chute.

**Type:** `str`

### `chute.uid`

The unique identifier for the chute.

**Type:** `str`

### `chute.readme`

The readme/documentation for the chute.

**Type:** `str`

### `chute.tagline`

The tagline for the chute.

**Type:** `str`

### `chute.image`

The image configuration for the chute.

**Type:** `str | Image`

### `chute.node_selector`

The hardware requirements for the chute.

**Type:** `NodeSelector | None`

### `chute.standard_template`

The standard template name if using a template.

**Type:** `str | None`

### `chute.cords`

List of cord endpoints registered with the chute.

**Type:** `list[Cord]`

### `chute.jobs`

List of jobs registered with the chute.

**Type:** `list[Job]`

## Methods

### `async chute.initialize()`

Initialize the chute by running all startup hooks. Called automatically when the chute starts in remote context.

```python
await chute.initialize()
```

## FastAPI Integration

Since `Chute` extends `FastAPI`, you can use all FastAPI features directly:

### Adding Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

@chute.on_startup()
async def setup_middleware(self):
    self.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
```

### Adding Custom Routes

```python
@chute.on_startup()
async def add_custom_routes(self):
    @self.get("/custom")
    async def custom_endpoint():
        return {"message": "Custom endpoint"}
```

### Using Dependencies

```python
from fastapi import Depends, HTTPException

async def verify_token(token: str):
    if token != "secret":
        raise HTTPException(401, "Invalid token")
    return token

@chute.cord(public_api_path="/protected")
async def protected_endpoint(self, token: str = Depends(verify_token)):
    return {"message": "Protected content"}
```

## Complete Example

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel, Field

# Define custom image
image = (
    Image(username="myuser", name="my-chute", tag="1.0")
    .from_base("parachutes/python:3.12")
    .run_command("pip install transformers torch")
)

# Define input/output schemas
class GenerationInput(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(100, ge=1, le=1000)

class GenerationOutput(BaseModel):
    text: str
    tokens_used: int

# Create chute
chute = Chute(
    username="myuser",
    name="text-generator",
    tagline="Generate text with transformers",
    readme="## Text Generator\n\nGenerates text from prompts.",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    concurrency=4,
    max_instances=3,
    shutdown_after_seconds=300,
    allow_external_egress=False
)

@chute.on_startup()
async def load_model(self):
    """Load model during startup."""
    from transformers import pipeline
    self.generator = pipeline("text-generation", model="gpt2", device=0)
    print("Model loaded!")

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
    minimal_input_schema=GenerationInput
)
async def generate(self, input_data: GenerationInput) -> GenerationOutput:
    """Generate text from a prompt."""
    result = self.generator(
        input_data.prompt,
        max_length=input_data.max_tokens
    )[0]["generated_text"]
    
    return GenerationOutput(
        text=result,
        tokens_used=len(result.split())
    )

@chute.cord(public_api_path="/health", public_api_method="GET")
async def health(self) -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": hasattr(self, "generator")
    }
```

## Best Practices

### 1. Use Appropriate Concurrency

```python
# For LLMs with continuous batching
chute = Chute(..., concurrency=64)

# For single-request models
chute = Chute(..., concurrency=1)
```

### 2. Set Reasonable Shutdown Timers

```python
# Development - short timeout
chute = Chute(..., shutdown_after_seconds=60)

# Production - longer timeout to avoid cold starts
chute = Chute(..., shutdown_after_seconds=300)
```

### 3. Use Type Hints and Schemas

```python
from pydantic import BaseModel

class MyInput(BaseModel):
    text: str

@chute.cord(
    public_api_path="/process",
    minimal_input_schema=MyInput
)
async def process(self, data: MyInput) -> dict:
    return {"result": data.text.upper()}
```

### 4. Handle Errors Gracefully

```python
from fastapi import HTTPException

@chute.cord(public_api_path="/generate")
async def generate(self, prompt: str):
    if not prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty")
    
    try:
        return await self.model.generate(prompt)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
```

## See Also

- **[Cord Decorator](/docs/sdk-reference/cord)** - Detailed cord documentation
- **[Job Decorator](/docs/sdk-reference/job)** - Job and server rental documentation
- **[Image Class](/docs/sdk-reference/image)** - Custom image building
- **[NodeSelector](/docs/sdk-reference/node-selector)** - Hardware requirements
- **[Templates](/docs/sdk-reference/templates)** - Pre-built templates
