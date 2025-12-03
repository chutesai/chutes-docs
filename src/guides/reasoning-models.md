# Reasoning Models Guide (DeepSeek R1)

DeepSeek R1 is a powerful open-source reasoning model that rivals proprietary models like OpenAI's o1. This guide shows you how to deploy DeepSeek R1 on Chutes using the SGLang template, optimized for high-performance reasoning tasks.

## Overview

DeepSeek R1 is a "reasoning model", meaning it is designed to "think" before it answers. This manifests as a chain-of-thought (CoT) process where the model explores the problem space, breaks down complex queries, and self-corrects before generating a final response.

Key requirements for deploying DeepSeek R1:
- **Large Context Window**: Reasoning traces can be long, requiring support for large context lengths (e.g., 65k-128k tokens).
- **High VRAM**: The full 671B parameter model (even quantized) requires significant GPU memory (multiple H100s/H200s).
- **Optimized Serving**: SGLang is recommended for its efficient handling of structured generation and long contexts.

## Quick Start: DeepSeek R1 Distill (Recommended)

For most use cases, the distilled versions of DeepSeek R1 (based on Llama 3 or Qwen 2.5) offer an excellent balance of performance and cost. They can often run on single GPUs.

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    readme="DeepSeek R1 Distill Llama 8B - Efficient Reasoning",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    revision="main",
    concurrency=16,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24, # Fits comfortably on A10g, A100, etc.
    ),
    engine_args={
        "max_model_len": 32768, # Reasoning models need context!
        "enable_prefix_caching": True,
    }
)
```

## Advanced: Full DeepSeek R1 (671B)

To deploy the full DeepSeek R1 model, you will need a multi-node or high-end multi-GPU setup. Chutes makes this accessible via the `sglang` template.

### Configuration

The full model is massive. We recommend using `chutes/sglang` images which are highly optimized for this workload.

```python
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

# Helper to configure environment for multi-node communication
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="myuser",
    readme="## DeepSeek R1 (Full 671B)\n\nState-of-the-art open reasoning model.",
    model_name="deepseek-ai/DeepSeek-R1",
    
    # Use a recent SGLang image for best R1 support
    image="chutes/sglang:0.4.6.post5b",
    
    concurrency=24,
    
    # Hardware Requirements
    node_selector=NodeSelector(
        gpu_count=8,           # Requires 8 GPUs
        min_vram_gb_per_gpu=140, # H200s or H100s with high memory usage
        include=["h200"],      # Specifically target H200s for best performance
    ),
    
    # SGLang Engine Arguments
    engine_args=(
        "--trust-remote-code "
        "--revision f7361cd9ff99396dbf6bd644ad846015e59ed4fc " # Pin a known good revision
        "--tp-size 8 "         # Tensor Parallelism across 8 GPUs
        "--context-length 65536 " # Large context for reasoning traces
        "--mem-fraction-static 0.90 " # Optimize memory usage
    ),
)
```

### Deployment

Save the above code to `deepseek_r1.py` and deploy:

```bash
chutes deploy deepseek_r1:chute
```

*Note: This deployment uses high-end hardware (8x H200s). Ensure your account has sufficient limits and balance.*

## Using Reasoning Models

When interacting with reasoning models, the "thinking process" is often returned as part of the output, enclosed in specific tags (e.g., `<think>...</think>`).

### Example Request

```python
import openai

client = openai.OpenAI(
    base_url="https://myuser-deepseek-r1.chutes.ai/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "How many Rs are in the word strawberry?"}
    ],
    temperature=0.6,
)

content = response.choices[0].message.content
print(content)
```

**Output Structure:**

```text
<think>
The user is asking for the count of the letter 'r' in "strawberry".
1. S-t-r-a-w-b-e-r-r-y
2. Let's count them:
   - s
   - t
   - r (1)
   - a
   - w
   - b
   - e
   - r (2)
   - r (3)
   - y
3. There are 3 Rs.
</think>

There are 3 Rs in the word "strawberry".
```

## Best Practices

1.  **Prompting**: Reasoning models respond well to simple, direct prompts. You often don't need complex "Chain of Thought" prompting strategies because the model does this natively.
2.  **Temperature**: Keep temperature slightly higher (0.5 - 0.7) than standard code models (0.0) to allow the model to explore different reasoning paths, but not too high to avoid incoherence.
3.  **Context Management**: The `<think>` traces consume tokens. Ensure your `max_model_len` / `context_length` is sufficient (e.g., 32k+) to accommodate long reasoning chains plus the final answer.
4.  **Streaming**: Always use `stream=True` for a better user experience, as the initial "thinking" phase can take several seconds before the final answer begins to appear.

## Troubleshooting

*   **OOM (Out of Memory)**: If the chute fails to start, try reducing `max_model_len` or `max_num_seqs` in `engine_args`. For the full 671B model, ensure you are targeting 8x80GB (A100/H100) or 8x141GB (H200) nodes.
*   **Slow "Time to First Token"**: This is normal for reasoning models as they generate internal thought tokens before producing visible output.

