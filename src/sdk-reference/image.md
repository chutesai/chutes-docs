# Image API Reference

The `Image` class is used to build custom Docker images for Chutes applications. This reference covers all methods, configuration options, and best practices for creating optimized container images.

## Class Definition

```python
from chutes.image import Image

image = Image(
    username: str,
    name: str,
    tag: str,
    readme: str = ""
)
```

## Constructor Parameters

### Required Parameters

#### `username: str`

The username or organization name for the image.

**Example:**

```python
image = Image(username="mycompany", name="custom-ai", tag="1.0")
```

**Rules:**

- Must match pattern `^[a-z0-9][a-z0-9-_\.]*$`
- Should match your Chutes username

#### `name: str`

The name of the Docker image.

**Example:**

```python
image = Image(username="mycompany", name="text-processor", tag="1.0")
```

**Rules:**

- Must match pattern `^[a-z0-9][a-z0-9-_\.]*$`
- Should be descriptive of the image purpose

#### `tag: str`

Version tag for the image.

**Examples:**

```python
# Version tag
image = Image(username="mycompany", name="ai-model", tag="1.0.0")

# Development tag
image = Image(username="mycompany", name="ai-model", tag="dev")
```

**Best Practices:**

- Use semantic versioning (1.0.0, 1.1.0, etc.)
- Use descriptive tags for different environments
- Avoid using "latest" in production

### Optional Parameters

#### `readme: str = ""`

Documentation for the image in Markdown format.

**Example:**

```python
readme = """
# Custom AI Processing Image

This image contains optimized libraries for AI text processing.

## Features
- PyTorch 2.0 with CUDA support
- Transformers library
- Optimized for GPU inference
"""

image = Image(
    username="mycompany",
    name="ai-processor",
    tag="1.0.0",
    readme=readme
)
```

## Default Base Image

By default, images use `parachutes/python:3.12` as the base image, which includes:

- CUDA 12.x support
- Python 3.12
- OpenCL libraries
- Common system dependencies

**We highly recommend using this base image** to avoid dependency issues.

## Methods

### `.from_base(base_image: str)`

Replace the base image.

**Signature:**

```python
def from_base(self, base_image: str) -> Image
```

**Examples:**

```python
# Use recommended Chutes base image (default)
image = Image("myuser", "myapp", "1.0").from_base("parachutes/python:3.12")

# Use NVIDIA CUDA base images
image = Image("myuser", "myapp", "1.0").from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")

# Use Python base images
image = Image("myuser", "myapp", "1.0").from_base("python:3.11-slim")
```

**Choosing Base Images:**

- **parachutes/python:3.12**: Recommended for most use cases
- **nvidia/cuda:\***: For GPU-accelerated applications needing specific CUDA versions
- **python:3.11-slim**: Lightweight, CPU-only workloads

### `.run_command(command: str)`

Execute shell commands during image build.

**Signature:**

```python
def run_command(self, command: str) -> Image
```

**Examples:**

```python
# Install Python packages
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .run_command("pip install torch transformers accelerate")
)

# Multiple commands in one call
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .run_command("""
        pip install --upgrade pip &&
        pip install torch transformers &&
        pip install accelerate datasets
    """)
)

# Install from requirements file
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("requirements.txt", "/tmp/requirements.txt")
    .run_command("pip install -r /tmp/requirements.txt")
)
```

### `.add(source: str, dest: str)`

Add files from the build context to the image.

**Signature:**

```python
def add(self, source: str, dest: str) -> Image
```

**Examples:**

```python
# Add single file
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("requirements.txt", "/app/requirements.txt")
)

# Add directory
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("src/", "/app/src/")
)

# Add multiple files
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("requirements.txt", "/app/requirements.txt")
    .add("config.yaml", "/app/config.yaml")
    .add("src/", "/app/src/")
)
```

**Best Practices:**

```python
# Add requirements first for better caching
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("requirements.txt", "/tmp/requirements.txt")  # Add early
    .run_command("pip install -r /tmp/requirements.txt")  # Install deps
    .add("src/", "/app/src/")  # Add code last (changes frequently)
)
```

### `.with_env(key: str, value: str)`

Set environment variables in the image.

**Signature:**

```python
def with_env(self, key: str, value: str) -> Image
```

**Examples:**

```python
# Basic environment variables
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .with_env("PYTHONPATH", "/app")
    .with_env("PYTHONUNBUFFERED", "1")
)

# Model cache configuration
image = (
    Image("myuser", "ai-app", "1.0")
    .from_base("parachutes/python:3.12")
    .with_env("TRANSFORMERS_CACHE", "/opt/models")
    .with_env("HF_HOME", "/opt/huggingface")
    .with_env("TORCH_HOME", "/opt/torch")
)

# Application configuration
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .with_env("APP_ENV", "production")
    .with_env("LOG_LEVEL", "INFO")
)
```

**Common Environment Variables:**

```python
# Python optimization
image = image.with_env("PYTHONOPTIMIZE", "2")
image = image.with_env("PYTHONDONTWRITEBYTECODE", "1")
image = image.with_env("PYTHONUNBUFFERED", "1")

# PyTorch optimizations
image = image.with_env("TORCH_BACKENDS_CUDNN_BENCHMARK", "1")
```

### `.set_workdir(directory: str)`

Set the working directory for the container.

**Signature:**

```python
def set_workdir(self, directory: str) -> Image
```

**Examples:**

```python
# Set working directory
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .set_workdir("/app")
    .add("src/", "/app/src/")
)

# Multiple working directories for different stages
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .set_workdir("/tmp")
    .add("requirements.txt", "requirements.txt")
    .run_command("pip install -r requirements.txt")
    .set_workdir("/app")
    .add("src/", "src/")
)
```

### `.set_user(user: str)`

Set the user for running commands and the container.

**Signature:**

```python
def set_user(self, user: str) -> Image
```

**Examples:**

```python
# Create and use non-root user
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .run_command("useradd -m -u 1000 appuser")
    .run_command("mkdir -p /app && chown appuser:appuser /app")
    .set_user("appuser")
    .set_workdir("/app")
)

# Use existing user
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("ubuntu:22.04")
    .set_user("nobody")
)
```

### `.apt_install(package: str | List[str])`

Install system packages using apt.

**Signature:**

```python
def apt_install(self, package: str | List[str]) -> Image
```

**Examples:**

```python
# Install single package
image = image.apt_install("git")

# Install multiple packages
image = image.apt_install(["git", "curl", "wget", "ffmpeg"])
```

### `.apt_remove(package: str | List[str])`

Remove system packages using apt.

**Signature:**

```python
def apt_remove(self, package: str | List[str]) -> Image
```

**Example:**

```python
# Remove packages after use
image = (
    image
    .apt_install(["build-essential", "cmake"])
    .run_command("pip install some-package-that-needs-compilation")
    .apt_remove(["build-essential", "cmake"])
)
```

### `.with_python(version: str = "3.10.15")`

Install a specific version of Python from source.

**Signature:**

```python
def with_python(self, version: str = "3.10.15") -> Image
```

**Example:**

```python
# Install specific Python version
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("ubuntu:22.04")
    .with_python("3.11.5")
)
```

**Note:** This builds Python from source, which can be slow. Consider using `parachutes/python:3.12` as your base image instead.

### `.with_maintainer(maintainer: str)`

Set the maintainer for the image.

**Signature:**

```python
def with_maintainer(self, maintainer: str) -> Image
```

**Example:**

```python
image = image.with_maintainer("team@mycompany.com")
```

### `.with_entrypoint(*args)`

Set the container entrypoint.

**Signature:**

```python
def with_entrypoint(self, *args) -> Image
```

**Examples:**

```python
# Python module entrypoint
image = image.with_entrypoint("python", "-m", "myapp")

# Shell script entrypoint
image = (
    image
    .add("entrypoint.sh", "/entrypoint.sh")
    .run_command("chmod +x /entrypoint.sh")
    .with_entrypoint("/entrypoint.sh")
)
```

## Complete Examples

### Basic ML Image

```python
from chutes.image import Image

image = (
    Image(username="myuser", name="ml-app", tag="1.0")
    .from_base("parachutes/python:3.12")
    .run_command("pip install torch transformers accelerate")
    .add("requirements.txt", "/app/requirements.txt")
    .run_command("pip install -r /app/requirements.txt")
    .add("src/", "/app/src/")
    .set_workdir("/app")
    .with_env("PYTHONPATH", "/app")
)
```

### Optimized PyTorch Image

```python
image = (
    Image(username="myuser", name="pytorch-app", tag="1.0",
          readme="## PyTorch Application\nOptimized for GPU inference.")
    .from_base("parachutes/python:3.12")

    # System dependencies
    .apt_install(["git", "curl", "ffmpeg"])

    # Python packages
    .run_command("""
        pip install --upgrade pip &&
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 &&
        pip install transformers accelerate datasets tokenizers
    """)

    # Environment optimization
    .with_env("PYTHONUNBUFFERED", "1")
    .with_env("TRANSFORMERS_CACHE", "/opt/models")
    .with_env("TORCH_BACKENDS_CUDNN_BENCHMARK", "1")

    # Application code
    .add("requirements.txt", "/app/requirements.txt")
    .run_command("pip install -r /app/requirements.txt")
    .add("src/", "/app/src/")
    .set_workdir("/app")
)
```

### Image with System Dependencies

```python
image = (
    Image(username="myuser", name="audio-processor", tag="1.0")
    .from_base("parachutes/python:3.12")

    # Audio processing dependencies
    .apt_install([
        "ffmpeg",
        "libsndfile1",
        "libportaudio2",
        "libsox-fmt-all"
    ])

    # Python audio libraries
    .run_command("""
        pip install soundfile librosa pydub torchaudio
    """)

    .add("src/", "/app/src/")
    .set_workdir("/app")
)
```

## Layer Caching Best Practices

For faster builds, order your directives from least to most frequently changing:

```python
# Good: Optimal layer ordering
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")

    # 1. System packages (rarely change)
    .apt_install(["git", "curl"])

    # 2. Python dependencies from requirements (change occasionally)
    .add("requirements.txt", "/tmp/requirements.txt")
    .run_command("pip install -r /tmp/requirements.txt")

    # 3. Application code (changes frequently)
    .add("src/", "/app/src/")

    .set_workdir("/app")
)

# Bad: Frequent changes early invalidate cache
image = (
    Image("myuser", "myapp", "1.0")
    .from_base("parachutes/python:3.12")
    .add("src/", "/app/src/")  # Changes often - invalidates all later layers!
    .apt_install(["git", "curl"])
    .run_command("pip install torch")
)
```

## Combining Commands

Combine related commands into single layers to reduce image size:

```python
# Good: Single layer with cleanup
image = image.run_command("""
    apt-get update &&
    apt-get install -y git curl &&
    rm -rf /var/lib/apt/lists/*
""")

# Less optimal: Multiple layers
image = (
    image
    .run_command("apt-get update")
    .run_command("apt-get install -y git curl")
    .run_command("rm -rf /var/lib/apt/lists/*")  # Cleanup in separate layer doesn't reduce size
)
```

## Properties

### `image.uid`

The unique identifier for the image.

**Type:** `str`

### `image.name`

The name of the image.

**Type:** `str`

### `image.tag`

The tag/version of the image.

**Type:** `str`

### `image.readme`

The documentation for the image.

**Type:** `str`

### `image.username`

The username/organization for the image.

**Type:** `str`

## See Also

- **[Chute Class](/docs/sdk-reference/chute)** - Using images with chutes
- **[Building Images](/docs/cli/build)** - CLI build commands
- **[Templates](/docs/sdk-reference/templates)** - Pre-built image templates
