# NodeSelector API Reference

The `NodeSelector` class specifies hardware requirements for Chutes deployments. This reference covers all configuration options, GPU types, and best practices for optimal resource allocation.

## Class Definition

```python
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count: int = 1,
    min_vram_gb_per_gpu: int = 16,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
)
```

## Parameters

### `gpu_count: int = 1`

Number of GPUs required for the deployment.

**Constraints:** 1-8 GPUs

**Examples:**

```python
# Single GPU (default)
node_selector = NodeSelector(gpu_count=1)

# Multiple GPUs for large models
node_selector = NodeSelector(gpu_count=4)

# Maximum supported GPUs
node_selector = NodeSelector(gpu_count=8)
```

**Use Cases:**

| GPU Count | Use Case |
|-----------|----------|
| 1 | Standard AI models (BERT, GPT-2, 7B LLMs) |
| 2-4 | Larger language models (13B-30B parameters) |
| 4-8 | Very large models (70B+ parameters) |

### `min_vram_gb_per_gpu: int = 16`

Minimum VRAM (Video RAM) required per GPU in gigabytes.

**Constraints:** 16-140 GB

**Examples:**

```python
# Default minimum (suitable for most models)
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16
)

# Medium models requiring more VRAM
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)

# Large models
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=48
)

# Ultra-large models (H100 80GB required)
node_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=80
)
```

**VRAM Requirements by Model Size:**

| Model Size | Min VRAM | Example Models |
|------------|----------|----------------|
| 1-3B params | 16GB | DistilBERT, GPT-2 |
| 7B params | 24GB | Llama-2-7B, Mistral-7B |
| 13B params | 32-40GB | Llama-2-13B |
| 30B params | 48GB | CodeLlama-34B |
| 70B+ params | 80GB+ | Llama-2-70B, DeepSeek-R1 |

### `include: Optional[List[str]] = None`

List of GPU types to include in selection. Only these GPU types will be considered.

**Examples:**

```python
# Only high-end GPUs
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"]
)

# Cost-effective options
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48,
    include=["l40", "a6000"]
)

# H100 only for maximum performance
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)
```

### `exclude: Optional[List[str]] = None`

List of GPU types to exclude from selection.

**Examples:**

```python
# Avoid older GPUs
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["t4"]
)

# Cost optimization - exclude expensive GPUs
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["h100", "a100-80gb"]
)
```

## Available GPU Types

### High-Performance GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| `h100` | 80GB | Latest Hopper architecture, best performance |
| `h200` | 141GB | Hopper with HBM3e, maximum memory |
| `a100-80gb` | 80GB | Ampere, excellent for training/inference |
| `a100` | 40GB | Ampere, high performance tier |

### Professional GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| `l40` | 48GB | Ada Lovelace, good balance of cost/performance |
| `a6000` | 48GB | Professional-grade, good for development |
| `a5000` | 24GB | Professional-grade, medium workloads |
| `a4000` | 16GB | Entry professional GPU |

### Consumer/Entry GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| `rtx4090` | 24GB | Consumer, cost-effective |
| `rtx3090` | 24GB | Previous gen consumer |
| `a10` | 24GB | Good for smaller models |
| `t4` | 16GB | Entry-level, inference-focused |

### AMD GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| `mi300x` | 192GB | AMD Instinct, very high memory |

## Common Selection Patterns

### Cost-Optimized

```python
# Small models - minimize cost
budget_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["t4", "a4000", "a10"]
)

# Medium models - balance cost/performance
balanced_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a5000", "rtx4090"],
    exclude=["h100", "a100-80gb"]
)
```

### Performance-Optimized

```python
# Maximum performance
performance_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100-80gb"]
)

# High throughput serving
throughput_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=48,
    include=["l40", "a100"]
)
```

### Model-Specific

```python
# 7B parameter models (e.g., Mistral-7B, Llama-2-7B)
llm_7b_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a5000", "rtx4090"]
)

# 13B parameter models
llm_13b_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=40,
    include=["l40", "a100", "a6000"]
)

# 70B parameter models
llm_70b_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100-80gb"]
)

# DeepSeek-R1 (671B parameters)
deepseek_selector = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=141,
    include=["h200"]
)
```

### Image Generation

```python
# Stable Diffusion / SDXL
diffusion_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a5000", "rtx4090"]
)

# FLUX models
flux_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48,
    include=["l40", "a6000", "a100"]
)
```

## Integration Examples

### With Chute Definition

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a100"]
)

chute = Chute(
    username="myuser",
    name="my-model-server",
    image=Image(username="myuser", name="my-image", tag="1.0"),
    node_selector=node_selector
)
```

### With Templates

```python
from chutes.chute.template import build_vllm_chute
from chutes.chute import NodeSelector

chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24,
        include=["l40", "a5000"]
    )
)
```

### Dynamic Selection Based on Model

```python
def get_node_selector(model_size: str) -> NodeSelector:
    """Get appropriate NodeSelector based on model size."""
    
    configs = {
        "small": {  # < 3B parameters
            "gpu_count": 1,
            "min_vram_gb_per_gpu": 16
        },
        "medium": {  # 7-13B parameters
            "gpu_count": 1,
            "min_vram_gb_per_gpu": 32,
            "exclude": ["t4"]
        },
        "large": {  # 30-70B parameters
            "gpu_count": 2,
            "min_vram_gb_per_gpu": 48,
            "include": ["a100", "l40", "h100"]
        },
        "xlarge": {  # 70B+ parameters
            "gpu_count": 4,
            "min_vram_gb_per_gpu": 80,
            "include": ["h100", "a100-80gb"]
        }
    }
    
    return NodeSelector(**configs.get(model_size, configs["medium"]))
```

## Common Issues and Solutions

### "No available nodes match your requirements"

**Solution 1:** Broaden your requirements

```python
# Too restrictive
strict_selector = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)

# More flexible
flexible_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=48,
    include=["h100", "a100", "l40"]
)
```

**Solution 2:** Reduce GPU count

```python
# Try multiple smaller GPUs
multi_gpu = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40
)
```

### "Out of memory" errors

Increase VRAM requirements:

```python
# Increase min_vram_gb_per_gpu
higher_vram = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48  # Increased from 24
)
```

## Best Practices

### 1. Right-Size Your Requirements

Don't over-provision - it wastes resources and costs more:

```python
# Bad - wastes resources for a 7B model
oversized = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80
)

# Good - matches actual needs
rightsized = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)
```

### 2. Use Include/Exclude Wisely

```python
# Be specific when you have known requirements
specific_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48,
    include=["l40", "a6000"]  # Known compatible GPUs
)

# Exclude known incompatible GPUs
compatible_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["t4"]  # Known to be too slow for your use case
)
```

### 3. Development vs Production

```python
# Development - prioritize cost
dev_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["t4", "a4000"]
)

# Production - prioritize performance
prod_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=48,
    include=["l40", "a100"],
    exclude=["t4", "a4000"]
)
```

## Summary

The NodeSelector provides control over GPU hardware selection with four parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gpu_count` | 1 | 1-8 | Number of GPUs |
| `min_vram_gb_per_gpu` | 16 | 16-140 | Minimum VRAM per GPU |
| `include` | None | List[str] | Whitelist GPU types |
| `exclude` | None | List[str] | Blacklist GPU types |

Start with minimum requirements and adjust based on performance needs and availability.

## See Also

- **[Chute Class](/docs/sdk-reference/chute)** - Using NodeSelector with chutes
- **[Templates](/docs/sdk-reference/templates)** - Pre-built templates with NodeSelector
- **[Cost Optimization](/docs/guides/cost-optimization)** - GPU selection for cost efficiency
