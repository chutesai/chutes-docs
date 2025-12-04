# SDK Reference

Complete SDK reference for the Chutes Python SDK. Each page documents the classes, functions, decorators, and methods available.

## Core Classes

- **[Chute Class](/docs/sdk-reference/chute)** - The main class for defining AI applications
- **[Cord Decorator](/docs/sdk-reference/cord)** - HTTP API endpoint decorator
- **[Job Decorator](/docs/sdk-reference/job)** - Long-running jobs and server rentals
- **[Image Class](/docs/sdk-reference/image)** - Docker image building
- **[NodeSelector Class](/docs/sdk-reference/node-selector)** - Hardware requirements

## Templates

- **[Template Functions](/docs/sdk-reference/templates)** - Pre-built templates for vLLM, SGLang, Diffusion, and Embeddings

## Quick Links

| Class | Import | Purpose |
|-------|--------|---------|
| `Chute` | `from chutes.chute import Chute` | Define AI applications |
| `NodeSelector` | `from chutes.chute import NodeSelector` | Specify GPU requirements |
| `Image` | `from chutes.image import Image` | Build custom images |
| `Port` | `from chutes.chute.job import Port` | Define job network ports |
| `build_vllm_chute` | `from chutes.chute.template import build_vllm_chute` | vLLM template |

## Reference Format

Each API reference includes:

- Class/function signature
- Parameter descriptions with types and defaults
- Usage examples
- Best practices
