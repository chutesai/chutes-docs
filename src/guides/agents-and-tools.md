# Function Calling, Agents, and Tool Use

This guide demonstrates how to build advanced AI applications using **function calling** (tool use) and **autonomous agents** on the Chutes platform. You'll learn how to enable models to interact with external tools, databases, and APIs.

## Overview

Chutes supports function calling through its optimized serving templates (vLLM and SGLang), enabling:

- **Structured Data Extraction**: Get JSON outputs guaranteed to match a schema
- **Tool Execution**: Allow models to call Python functions
- **Agentic Workflows**: Build multi-step reasoning agents
- **External Integrations**: Connect LLMs to APIs, databases, and the web

## Quick Start: Enabling Function Calling

Use the `vLLM` template with specific arguments to enable tool support.

### 1. Deploy a Tool-Compatible Model

Models like **Mistral**, **Llama 3**, and **Qwen** have excellent function calling capabilities.

```python
# deploy_agent_chute.py
from chutes.chute import NodeSelector
from chutes.chute.template import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    ),
    engine_args={
        "enable_auto_tool_choice": True,  # Enable tool parsing
        "tool_call_parser": "mistral",    # Use specific parser (or "llama3_json")
        "max_model_len": 8192
    }
)
```

Deploy this chute:
```bash
chutes deploy deploy_agent_chute:chute --wait
```

## Building a Simple Agent

Here is a complete example of a Python client interacting with your deployed chute to execute tools.

### The Client Code

```python
import openai
import json
import math

# 1. Define the tools
def calculate_square_root(x: float) -> float:
    """Calculates the square root of a number."""
    return math.sqrt(x)

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Mock response
    return json.dumps({"location": location, "temperature": "72F", "condition": "Sunny"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_square_root",
            "description": "Calculates the square root of a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "The number to calculate the root of"}
                },
                "required": ["x"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                },
                "required": ["location"]
            }
        }
    }
]

# 2. Initialize Client
client = openai.OpenAI(
    base_url="https://myuser-mistral-7b.chutes.ai/v1",
    api_key="your-api-key"
)

# 3. Chat Loop with Tool Execution
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What is the square root of 144 and what's the weather in Miami?"}
]

# First call: Model decides to call tools
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    # Append the model's response (containing tool calls) to history
    messages.append(response_message)

    # Execute each tool call
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"🛠️ Executing {function_name} with {function_args}...")
        
        if function_name == "calculate_square_root":
            result = str(calculate_square_root(**function_args))
        elif function_name == "get_weather":
            result = get_weather(**function_args)
        else:
            result = "Error: Unknown function"

        # Append tool result to history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": result
        })

    # Second call: Model uses tool results to generate final answer
    final_response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages
    )
    
    print(f"🤖 Agent: {final_response.choices[0].message.content}")
```

## Structured Output (JSON Mode)

Sometimes you don't need to execute a function, but just want **guaranteed JSON output**.

```python
# Define the schema you want
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "score": {"type": "number"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "score", "keywords"]
}

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "user", "content": "Analyze this review: 'The product is decent but expensive.'"}
    ],
    # Force JSON mode
    response_format={"type": "json_object"},
    # Optionally pass schema in system prompt or specific guided decoding parameters if using SGLang
)

print(response.choices[0].message.content)
# Output: {"sentiment": "neutral", "score": 0.5, "keywords": ["decent", "expensive"]}
```

## Advanced: SGLang for High-Speed Agents

For complex agentic workflows requiring **constrained generation** (e.g., "Output must be valid SQL"), SGLang is superior.

### 1. Deploy SGLang Chute

```python
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    node_selector=NodeSelector(gpu_count=1),
    engine_args={
        "disable_flashinfer": False
    }
)
```

### 2. Using Regex Constraints (Client-Side)

SGLang supports `extra_body` parameters for regex constraints:

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "What is the IP address of localhost?"}],
    extra_body={
        "regex": r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    }
)
print(response.choices[0].message.content)
# Guaranteed to be a valid IP format
```

## Building a RAG Agent

Combine **Function Calling** with **Chutes Embeddings** for a RAG (Retrieval Augmented Generation) agent.

### Architecture

1.  **Vector Store**: Stores your documents (e.g., Qdrant/pgvector running in a separate Chute or externally).
2.  **Embedding Chute**: TEI template for generating query embeddings.
3.  **Agent Chute**: vLLM/SGLang model with a `search_knowledge_base` tool.

### Implementation Sketch

```python
def search_knowledge_base(query: str):
    """Tool exposed to the LLM."""
    # 1. Embed query using Chutes TEI endpoint
    embedding = requests.post(
        "https://myuser-embeddings.chutes.ai/embed", 
        json={"inputs": query}
    ).json()
    
    # 2. Search vector DB
    results = vector_db.search(embedding)
    
    # 3. Return context
    return json.dumps(results)

# ... Add this tool to the tools list in the Client Code example above ...
```

## Best Practices for Agents

1.  **System Prompts**: Clearly define the agent's persona and constraints.
    *   *Bad:* "You are a bot."
    *   *Good:* "You are a data analysis assistant. You have access to a SQL database. Always verify schemas before querying."
2.  **Tool Descriptions**: Models rely heavily on tool descriptions. Be verbose and precise.
3.  **Error Handling**: If a tool fails, feed the error message back to the model as a "tool" role message. The model can often self-correct.
4.  **Concurrency**: For agents that make parallel tool calls, use Python's `asyncio.gather` to execute them concurrently before responding to the model.

## Next Steps

- **[Embedding Service](/docs/examples/embeddings)** - Set up your RAG backend
- **[SGLang Template](/docs/templates/sglang)** - Advanced constrained generation
- **[vLLM Template](/docs/templates/vllm)** - High-performance tool serving

