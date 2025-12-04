# Cord Decorator API Reference

The `@chute.cord()` decorator is used to create HTTP API endpoints in Chutes applications. Cords are the primary way to expose functionality from your chute. This reference covers all parameters, patterns, and best practices.

## Decorator Signature

```python
@chute.cord(
    path: str = None,
    passthrough_path: str = None,
    passthrough: bool = False,
    passthrough_port: int = None,
    public_api_path: str = None,
    public_api_method: str = "POST",
    stream: bool = False,
    provision_timeout: int = 180,
    input_schema: Optional[Any] = None,
    minimal_input_schema: Optional[Any] = None,
    output_content_type: Optional[str] = None,
    output_schema: Optional[Dict] = None,
    **session_kwargs
)
```

## Parameters

### `public_api_path: str`

The URL path where the endpoint will be accessible via the public API.

**Format Rules:**

- Must start with `/`
- Must match pattern `^(/[a-z0-9_]+[a-z0-9-_]*)+$`
- Can include path parameters with `{parameter_name}` syntax
- Case-sensitive

**Examples:**

```python
# Simple path
@chute.cord(public_api_path="/generate")

# Path with parameter
@chute.cord(public_api_path="/users/{user_id}")

# Nested resource
@chute.cord(public_api_path="/models/{model_id}/generate")
```

### `public_api_method: str = "POST"`

The HTTP method for the public API endpoint.

**Supported Methods:**

- `GET` - Retrieve data
- `POST` - Create or process data (default)
- `PUT` - Update existing data
- `DELETE` - Remove data
- `PATCH` - Partial updates

**Examples:**

```python
# GET for data retrieval
@chute.cord(public_api_path="/models", public_api_method="GET")
async def list_models(self):
    return {"models": ["gpt-3.5", "gpt-4"]}

# POST for data processing (default)
@chute.cord(public_api_path="/generate", public_api_method="POST")
async def generate_text(self, prompt: str):
    return await self.model.generate(prompt)

# DELETE for removal
@chute.cord(public_api_path="/cache", public_api_method="DELETE")
async def clear_cache(self):
    self.cache.clear()
    return {"status": "cache cleared"}
```

### `path: str = None`

Internal path for the endpoint. Defaults to the function name if not specified.

### `stream: bool = False`

Enable streaming responses for real-time data transmission.

**When to Use Streaming:**

- Long-running text generation
- Real-time progress updates
- Token-by-token LLM output
- Large data processing

**Streaming Example:**

```python
from fastapi.responses import StreamingResponse
import json

@chute.cord(
    public_api_path="/stream_generate",
    public_api_method="POST",
    stream=True
)
async def stream_text_generation(self, prompt: str):
    async def generate_stream():
        async for token in self.model.stream_generate(prompt):
            data = {"token": token, "finished": False}
            yield f"data: {json.dumps(data)}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

### `input_schema: Optional[Any] = None`

Pydantic model for input validation and documentation.

**Benefits:**

- Automatic input validation
- Auto-generated API documentation
- Type safety
- Error handling

**Example:**

```python
from pydantic import BaseModel, Field

class TextGenerationInput(BaseModel):
    prompt: str = Field(..., description="Text prompt for generation")
    max_tokens: int = Field(100, ge=1, le=2000, description="Maximum tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
    input_schema=TextGenerationInput
)
async def generate_text(self, input_data: TextGenerationInput):
    return await self.model.generate(
        input_data.prompt,
        max_tokens=input_data.max_tokens,
        temperature=input_data.temperature
    )
```

### `minimal_input_schema: Optional[Any] = None`

Simplified schema for basic API documentation and testing. Useful when you have complex input but want simpler examples.

**Example:**

```python
class FullInput(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0

class SimpleInput(BaseModel):
    prompt: str = Field(..., description="Just the prompt for quick testing")

@chute.cord(
    public_api_path="/generate",
    input_schema=FullInput,
    minimal_input_schema=SimpleInput
)
async def generate_flexible(self, input_data: FullInput):
    return await self.model.generate(**input_data.dict())
```

### `output_content_type: Optional[str] = None`

The MIME type of the response content. Auto-detected for JSON/text, but should be specified for binary responses.

**Common Content Types:**

- `application/json` - JSON responses (auto-detected)
- `text/plain` - Plain text (auto-detected)
- `image/png`, `image/jpeg` - Images
- `audio/wav`, `audio/mpeg` - Audio files
- `text/event-stream` - Server-sent events

**Image Response Example:**

```python
from fastapi import Response

@chute.cord(
    public_api_path="/generate_image",
    public_api_method="POST",
    output_content_type="image/png"
)
async def generate_image(self, prompt: str) -> Response:
    image_data = await self.image_model.generate(prompt)

    return Response(
        content=image_data,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=generated.png"}
    )
```

**Audio Response Example:**

```python
@chute.cord(
    public_api_path="/text_to_speech",
    public_api_method="POST",
    output_content_type="audio/wav"
)
async def text_to_speech(self, text: str) -> Response:
    audio_data = await self.tts_model.synthesize(text)

    return Response(
        content=audio_data,
        media_type="audio/wav"
    )
```

### `output_schema: Optional[Dict] = None`

Schema for output validation and documentation. Auto-extracted from return type hints.

### `passthrough: bool = False`

Enable passthrough mode to forward requests to an underlying service.

**Use Case:** When you're running a service like vLLM or SGLang that has its own HTTP server, you can use passthrough to forward requests.

**Example:**

```python
@chute.cord(
    public_api_path="/v1/completions",
    public_api_method="POST",
    passthrough=True,
    passthrough_path="/v1/completions",
    passthrough_port=8000
)
async def completions(self, **kwargs):
    # Request is forwarded to localhost:8000/v1/completions
    pass
```

### `passthrough_path: str = None`

The path to forward requests to when using passthrough mode.

### `passthrough_port: int = None`

The port to forward requests to when using passthrough mode. Defaults to 8000.

### `provision_timeout: int = 180`

Timeout in seconds for waiting for the chute to provision. Default is 3 minutes.

## Function Patterns

### Simple Functions

```python
# Basic function with primitive parameters
@chute.cord(public_api_path="/simple")
async def simple_endpoint(self, text: str, number: int = 10):
    return {"text": text, "number": number}

# Function with optional parameters
@chute.cord(public_api_path="/optional")
async def optional_params(
    self,
    required_param: str,
    optional_param: str = None,
    default_param: int = 100
):
    return {
        "required": required_param,
        "optional": optional_param,
        "default": default_param
    }
```

### Schema-Based Functions

```python
from pydantic import BaseModel

class MyInput(BaseModel):
    text: str
    count: int = 1

class MyOutput(BaseModel):
    results: list[str]

@chute.cord(
    public_api_path="/process",
    input_schema=MyInput,
    output_schema=MyOutput
)
async def process_with_schemas(self, data: MyInput) -> MyOutput:
    results = [data.text] * data.count
    return MyOutput(results=results)
```

### File Responses

```python
from fastapi.responses import FileResponse

@chute.cord(
    public_api_path="/download",
    public_api_method="GET",
    output_content_type="application/pdf"
)
async def download_file(self) -> FileResponse:
    return FileResponse(
        "report.pdf",
        media_type="application/pdf",
        filename="report.pdf"
    )
```

## Error Handling

```python
from fastapi import HTTPException

@chute.cord(public_api_path="/generate")
async def generate_with_errors(self, prompt: str):
        # Validate input
        if not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if len(prompt) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Prompt too long (max 10,000 characters)"
            )

    try:
        result = await self.model.generate(prompt)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
```

## Complete Example

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel, Field
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import json

image = (
    Image(username="myuser", name="text-gen", tag="1.0")
    .from_base("parachutes/python:3.12")
    .run_command("pip install transformers torch")
)

chute = Chute(
    username="myuser",
    name="text-generator",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
    concurrency=4
)

class GenerationInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(100, ge=1, le=2000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class SimpleInput(BaseModel):
    prompt: str

@chute.on_startup()
async def load_model(self):
    from transformers import pipeline
    self.generator = pipeline("text-generation", model="gpt2", device=0)

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
    input_schema=GenerationInput,
    minimal_input_schema=SimpleInput
)
async def generate(self, params: GenerationInput) -> dict:
    """Generate text from a prompt."""
    result = self.generator(
        params.prompt,
        max_length=params.max_tokens,
        temperature=params.temperature
    )[0]["generated_text"]
    
    return {
        "generated_text": result,
        "tokens_used": len(result.split())
    }

@chute.cord(
    public_api_path="/stream",
    public_api_method="POST",
    stream=True
)
async def stream_generate(self, prompt: str):
    """Stream text generation token by token."""
    async def generate():
        # Simulated streaming
        words = prompt.split()
        for word in words:
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        yield f"data: {json.dumps({'finished': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@chute.cord(public_api_path="/health", public_api_method="GET")
async def health(self) -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": hasattr(self, "generator")
    }
```

## Best Practices

### 1. Use Descriptive Paths

   ```python
   # Good
   @chute.cord(public_api_path="/generate_text")
   @chute.cord(public_api_path="/analyze_sentiment")

   # Avoid
   @chute.cord(public_api_path="/api")
@chute.cord(public_api_path="/do")
   ```

### 2. Choose Appropriate Methods

   ```python
# GET for read-only operations
@chute.cord(public_api_path="/models", public_api_method="GET")

# POST for AI generation/processing
@chute.cord(public_api_path="/generate", public_api_method="POST")
   ```

### 3. Use Input Schemas for Validation

   ```python
from pydantic import BaseModel, Field

class ValidatedInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

@chute.cord(public_api_path="/generate", input_schema=ValidatedInput)
async def generate(self, params: ValidatedInput):
    # Input is automatically validated
    pass
```

### 4. Handle Errors Gracefully

   ```python
@chute.cord(public_api_path="/generate")
async def generate(self, prompt: str):
   if not prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty")
    
    try:
        return await self.model.generate(prompt)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
   ```

### 5. Use Streaming for Long Operations

   ```python
@chute.cord(public_api_path="/generate", stream=True)
async def stream_generate(self, prompt: str):
    async def stream():
        async for token in self.model.stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
   ```

## See Also

- **[Chute Class](/docs/sdk-reference/chute)** - Main chute documentation
- **[Job Decorator](/docs/sdk-reference/job)** - Background job documentation
- **[Streaming Guide](/docs/guides/streaming)** - Detailed streaming patterns
