# Templates API Reference

Chutes provides pre-built templates for common AI/ML frameworks and use cases. Templates are factory functions that create pre-configured `Chute` instances with optimized settings for specific AI frameworks.

## Overview

Templates provide:

- **Quick Setup**: Instant deployment of popular AI models
- **Best Practices**: Pre-configured optimization settings
- **Standard APIs**: OpenAI-compatible endpoints for LLMs
- **Customization**: Override any parameter as needed

## Available Templates

| Template | Use Case | Import |
|----------|----------|--------|
| `build_vllm_chute` | LLM serving with vLLM | `from chutes.chute.template import build_vllm_chute` |
| `build_sglang_chute` | LLM serving with SGLang | `from chutes.chute.template.sglang import build_sglang_chute` |
| `build_diffusion_chute` | Image generation | `from chutes.chute.template.diffusion import build_diffusion_chute` |
| `build_embedding_chute` | Text embeddings | `from chutes.chute.template.embedding import build_embedding_chute` |

## vLLM Template

### `build_vllm_chute()`

Create a chute optimized for vLLM (high-performance LLM serving) with OpenAI-compatible API endpoints.

**Import:**

```python
from chutes.chute.template import build_vllm_chute
```

**Signature:**

```python
def build_vllm_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = VLLM,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 64,
    engine_args: Dict[str, Any] = {},
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    allow_external_egress: bool = False
) -> Chute
```

**Parameters:**

- **`username`** - Your Chutes username (required)
- **`model_name`** - HuggingFace model identifier (required)
- **`node_selector`** - Hardware requirements (required)
- **`image`** - Custom vLLM image (defaults to built-in)
- **`tagline`** - Brief description
- **`readme`** - Detailed documentation
- **`concurrency`** - Max concurrent requests (default: 64)
- **`engine_args`** - vLLM engine configuration
- **`revision`** - Model revision
- **`max_instances`** - Max scaling instances (default: 1)
- **`scaling_threshold`** - Scaling trigger threshold (default: 0.75)
- **`shutdown_after_seconds`** - Idle shutdown time (default: 300)
- **`allow_external_egress`** - Allow external network access (default: False)

**Basic Example:**

```python
from chutes.chute.template import build_vllm_chute
from chutes.chute import NodeSelector

chute = build_vllm_chute(
    username="myuser",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    )
)
```

**Advanced Example:**

```python
from chutes.chute.template import build_vllm_chute
from chutes.chute import NodeSelector

chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-70b-chat-hf",
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=48,
        exclude=["l40", "a6000"]
    ),
    engine_args={
        "gpu_memory_utilization": 0.97,
        "max_model_len": 4096,
        "max_num_seqs": 8,
        "trust_remote_code": True,
        "tensor_parallel_size": 8
    },
    concurrency=8,
    max_instances=3
)
```

**Common vLLM Engine Arguments:**

```python
engine_args = {
    # Memory management
    "gpu_memory_utilization": 0.95,   # Use 95% of GPU memory
    "swap_space": 4,                   # GB of CPU swap space

    # Model configuration
    "max_model_len": 4096,             # Maximum sequence length
    "max_num_seqs": 256,               # Maximum concurrent sequences
    "trust_remote_code": False,        # Allow custom model code

    # Performance optimization
    "enable_prefix_caching": True,     # Cache prefixes for efficiency
    "use_v2_block_manager": True,      # Improved block manager

    # Quantization
    "quantization": None,              # e.g., "awq", "gptq", "fp8"
    "dtype": "auto",                   # Model data type

    # Distributed inference
    "tensor_parallel_size": 1,         # GPUs for tensor parallelism

    # Tokenizer
    "tokenizer_mode": "auto",          # Tokenizer mode
    
    # Mistral-specific
    "config_format": "mistral",        # For Mistral models
    "load_format": "mistral",
    "tool_call_parser": "mistral",
    "enable_auto_tool_choice": True
}
```

**Provided Endpoints:**

vLLM template provides OpenAI-compatible endpoints:

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `POST /v1/tokenize` - Tokenization
- `POST /v1/detokenize` - Detokenization
- `GET /v1/models` - List available models

## SGLang Template

### `build_sglang_chute()`

Create a chute optimized for SGLang (structured generation language serving).

**Import:**

```python
from chutes.chute.template.sglang import build_sglang_chute
```

**Signature:**

```python
def build_sglang_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = SGLANG,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 64,
    engine_args: Dict[str, Any] = {},
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    allow_external_egress: bool = False
) -> Chute
```

**Example:**

```python
from chutes.chute.template.sglang import build_sglang_chute
from chutes.chute import NodeSelector

chute = build_sglang_chute(
    username="myuser",
    model_name="deepseek-ai/DeepSeek-R1",
    node_selector=NodeSelector(
        gpu_count=8,
        include=["h200"],
        min_vram_gb_per_gpu=141
    ),
    engine_args={
        "host": "0.0.0.0",
        "port": 30000,
        "tp_size": 8,
        "trust_remote_code": True,
        "context_length": 65536,
        "mem_fraction_static": 0.8
    },
    concurrency=4
)
```

**Common SGLang Engine Arguments:**

```python
engine_args = {
    # Server configuration
    "host": "0.0.0.0",
    "port": 30000,

    # Model configuration
    "context_length": 4096,
    "trust_remote_code": True,

    # Performance
    "tp_size": 1,                    # Tensor parallelism
    "mem_fraction_static": 0.9,      # Static memory fraction
    "chunked_prefill_size": 512,

    # Features
    "enable_flashinfer": True
}
```

## Diffusion Template

### `build_diffusion_chute()`

Create a chute optimized for diffusion model inference (image generation).

**Import:**

```python
from chutes.chute.template.diffusion import build_diffusion_chute
```

**Example:**

```python
from chutes.chute.template.diffusion import build_diffusion_chute
from chutes.chute import NodeSelector

chute = build_diffusion_chute(
    username="myuser",
    model_name="black-forest-labs/FLUX.1-dev",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=48,
        include=["l40", "a100"]
    ),
    engine_args={
        "torch_dtype": "bfloat16",
        "guidance_scale": 3.5,
        "num_inference_steps": 28
    },
    concurrency=1  # Image generation is typically 1 concurrent request
)
```

**Generation Input Schema:**

```python
from pydantic import BaseModel, Field

class GenerationInput(BaseModel):
    prompt: str
    negative_prompt: str = ""
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=25, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
```

**Provided Endpoints:**

- `POST /generate` - Generate image from prompt

## Embedding Template

### `build_embedding_chute()`

Create a chute optimized for text embeddings using vLLM.

**Import:**

```python
from chutes.chute.template.embedding import build_embedding_chute
```

**Signature:**

```python
def build_embedding_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = VLLM,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 32,
    engine_args: Dict[str, Any] = {},
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    pooling_type: str = "auto",
    max_embed_len: int = 3072000,
    enable_chunked_processing: bool = True,
    allow_external_egress: bool = False
) -> Chute
```

**Example:**

```python
from chutes.chute.template.embedding import build_embedding_chute
from chutes.chute import NodeSelector

chute = build_embedding_chute(
    username="myuser",
    model_name="BAAI/bge-large-en-v1.5",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    pooling_type="auto",  # Auto-detect optimal pooling
    concurrency=32
)
```

**Pooling Types:**

- `"auto"` - Auto-detect based on model name
- `"MEAN"` - Mean pooling (E5, Jina models)
- `"CLS"` - CLS token pooling (BGE models)
- `"LAST"` - Last token pooling (GTE, Qwen models)

**Provided Endpoints:**

- `POST /v1/embeddings` - OpenAI-compatible embeddings endpoint

## Extending Templates

Templates can be extended with custom functionality:

```python
from chutes.chute.template import build_vllm_chute
from chutes.chute import NodeSelector

# Create base chute from template
chute = build_vllm_chute(
    username="myuser",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24)
)

# Add custom endpoint
@chute.cord(public_api_path="/summarize", public_api_method="POST")
async def summarize(self, text: str) -> dict:
    """Summarize text using the loaded model."""
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"

    # Use the template's built-in generation
    result = await self.generate(prompt=prompt, max_tokens=200)

    return {"summary": result}

# Add custom startup logic
@chute.on_startup(priority=90)  # Run after template initialization
async def custom_setup(self):
    """Custom initialization after model loads."""
    print("Custom setup complete!")
```

## Model-Specific Configurations

### Mistral Models

```python
    chute = build_vllm_chute(
    username="myuser",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
    engine_args={
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "tool_call_parser": "mistral",
        "enable_auto_tool_choice": True
    }
)
```

### Llama Models

```python
chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-70b-chat-hf",
        node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=48
    ),
        engine_args={
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.95,
        "tensor_parallel_size": 4
    }
    )
```

### DeepSeek Models

```python
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    model_name="deepseek-ai/DeepSeek-R1",
    node_selector=NodeSelector(
                gpu_count=8,
        min_vram_gb_per_gpu=141,
        include=["h200"]
            ),
    engine_args={
        "tp_size": 8,
        "trust_remote_code": True,
        "context_length": 65536
    }
    )
```

### FLUX Image Generation

```python
from chutes.chute.template.diffusion import build_diffusion_chute

chute = build_diffusion_chute(
    username="myuser",
    model_name="black-forest-labs/FLUX.1-dev",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=48
    ),
    engine_args={
        "torch_dtype": "bfloat16",
        "guidance_scale": 3.5,
        "num_inference_steps": 28
    }
)
```

## Best Practices

### 1. Choose the Right Template

   ```python
# For OpenAI-compatible LLM API
vllm_chute = build_vllm_chute(...)

# For structured generation and reasoning
sglang_chute = build_sglang_chute(...)

# For text embeddings
embedding_chute = build_embedding_chute(...)

   # For image generation
diffusion_chute = build_diffusion_chute(...)
   ```

### 2. Match Hardware to Model

   ```python
# 7B model - single GPU
node_selector = NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24)

# 70B model - multiple GPUs with tensor parallelism
node_selector = NodeSelector(gpu_count=4, min_vram_gb_per_gpu=48)
engine_args = {"tensor_parallel_size": 4}
   ```

### 3. Set Appropriate Concurrency

   ```python
# vLLM/SGLang with continuous batching - high concurrency
chute = build_vllm_chute(..., concurrency=64)

# Image generation - low concurrency
chute = build_diffusion_chute(..., concurrency=1)

# Embeddings - medium-high concurrency
chute = build_embedding_chute(..., concurrency=32)
   ```

### 4. Use Auto-Scaling for Production

   ```python
chute = build_vllm_chute(
    ...,
    max_instances=10,
    scaling_threshold=0.75,
    shutdown_after_seconds=300
)
```

## See Also

- **[Chute Class](/docs/sdk-reference/chute)** - Chute class reference
- **[NodeSelector](/docs/sdk-reference/node-selector)** - Hardware requirements
- **[vLLM Template Guide](/docs/templates/vllm)** - Detailed vLLM documentation
- **[SGLang Template Guide](/docs/templates/sglang)** - Detailed SGLang documentation
- **[Diffusion Template Guide](/docs/templates/diffusion)** - Image generation guide
