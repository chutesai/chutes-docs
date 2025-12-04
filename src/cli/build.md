# Building Images

The `chutes build` command creates Docker images for your chutes with all necessary dependencies and optimizations for the Chutes platform.

## Basic Build Command

### `chutes build`

Build a Docker image for your chute.

```bash
chutes build <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image for the image
- `--local`: Build locally instead of remotely (useful for testing/debugging)
- `--debug`: Enable debug logging
- `--include-cwd`: Include entire current directory in build context recursively
- `--wait`: Wait for remote build to complete and stream logs
- `--public`: Mark image as public/available to anyone

## Build Examples

### Basic Remote Build

```bash
# Build on Chutes infrastructure (recommended)
chutes build my_chute:chute --wait
```

**Benefits of Remote Building:**

- 🚀 Faster build times with powerful infrastructure
- 📦 Optimized caching and layer sharing
- 🔒 Secure build environment
- 💰 No local resource usage

### Local Development Build

```bash
# Build locally for testing and development
chutes build my_chute:chute --local --debug
```

**When to Use Local Builds:**

- 🧪 Quick development iterations
- 🔍 Debugging build issues
- 🌐 Limited internet connectivity
- 🔒 Sensitive code that shouldn't leave your machine

### Production Build with Assets

```bash
# Build with logo and make public
chutes build my_chute:chute --logo ./assets/logo.png --public --wait
```

## Build Process

### What Happens During Build

1. **Code Analysis**: Chutes analyzes your Python code and image directives
2. **Context Packaging**: Build context files are packaged and uploaded
3. **Image Creation**: Dockerfile is generated from your Image definition
4. **Dependency Installation**: Python packages and system dependencies installed
5. **Validation**: Image is validated for compatibility

### Build Stages

```bash
# Example build output
Building chute: my_chute:chute
✓ Analyzing code structure
✓ Packaging build context
✓ Uploading to build server
✓ Building image layers
✓ Installing dependencies
✓ Pushing to registry

Build completed successfully!
Image ID: img_abc123def456
```

### Build Context

When building remotely, the CLI will:

1. Collect all files referenced in your `Image` directives
2. Show you which files will be uploaded
3. Ask for confirmation before uploading
4. Package and send to the build server

```bash
Found 15 files to include in build context -- these will be uploaded for remote builds!
 requirements.txt
 src/main.py
 src/utils.py
 ...
Confirm submitting build context? (y/n)
```

## Image Definition

Images are defined in Python using the `Image` class:

```python
from chutes.image import Image

image = (
    Image(username="myuser", name="my-chute", tag="1.0")
    .from_base("parachutes/python:3.12")
    .run_command("apt-get update && apt-get install -y git")
    .add("requirements.txt", "/app/requirements.txt")
    .run_command("pip install -r /app/requirements.txt")
    .add("src/", "/app/src/")
)
```

### Recommended Base Image

We **highly recommend** starting with our base image to avoid dependency issues:

```python
.from_base("parachutes/python:3.12")
```

This base image includes:

- CUDA 12.x installation
- Python 3.12
- OpenCL libraries
- Common ML dependencies

### Build Context Optimization

Organize your directives for optimal caching:

```python
# Good: Stable operations first, frequently changing code last
image = (
    Image(username="myuser", name="my-app", tag="1.0")
    .from_base("parachutes/python:3.12")

    # System deps (rarely change)
    .run_command("apt-get update && apt-get install -y git curl")

    # Python deps (change occasionally)
    .add("requirements.txt", "/app/requirements.txt")
    .run_command("pip install -r /app/requirements.txt")

    # Application code (changes frequently)
    .add("src/", "/app/src/")
)
```

## Including Files

### Automatic Context Detection

The build system automatically detects files referenced in your `Image.add()` directives:

```python
image = (
    Image(...)
    .add("requirements.txt", "/app/requirements.txt")  # Only this file included
    .add("src/", "/app/src/")  # This directory included
)
```

### Including Entire Directory

Use `--include-cwd` to include the entire current directory:

```bash
chutes build my_chute:chute --include-cwd --wait
```

This is useful when your code has implicit dependencies not captured in the Image definition.

## Troubleshooting Builds

### Common Build Issues

**Build fails with dependency errors?**

```bash
# Build with debug to see full output
chutes build my_chute:chute --local --debug

# Check your requirements.txt versions are compatible
cat requirements.txt
```

**Image already exists?**

```bash
# Check existing images
chutes images list --name my-chute

# Delete old image if needed
chutes images delete my-chute:1.0
```

**Build takes too long?**

- Use remote building (usually faster): `chutes build my_chute:chute --wait`
- Optimize Docker layers in your Image definition
- Put stable dependencies (like torch) before frequently changing code

**Permission errors (local build)?**

```bash
# Check Docker daemon is running
sudo systemctl status docker

# Check file permissions
ls -la
```

### Debug Commands

```bash
# Inspect generated Dockerfile
python -c "from my_chute import chute; print(chute.image)"

# Check image exists after build
chutes images list --name my-chute
chutes images get my-chute
```

## Build Strategies

### Development Workflow

```bash
# Fast iteration during development with local builds
chutes build my_chute:chute --local

# Test the built image locally
docker run --rm -it -p 8000:8000 my_chute:1.0 chutes run my_chute:chute --dev

# Once stable, build remotely
chutes build my_chute:chute --wait
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install Chutes
        run: pip install chutes

      - name: Configure Chutes
        env:
          CHUTES_CONFIG: ${{ secrets.CHUTES_CONFIG }}
        run: |
          mkdir -p ~/.chutes
          echo "$CHUTES_CONFIG" > ~/.chutes/config.ini

      - name: Build Image
        run: chutes build my_app:chute --wait
```

### Production Builds

```bash
#!/bin/bash
set -e

echo "Building production image..."

# 1. Ensure clean workspace
git status --porcelain
[ -z "$(git status --porcelain)" ] || { echo "Uncommitted changes found"; exit 1; }

# 2. Run tests
python -m pytest tests/

# 3. Build image
chutes build my_chute:chute --wait

# 4. Deploy
chutes deploy my_chute:chute --accept-fee

echo "Production build and deploy completed!"
```

## Best Practices

### 1. Pin Dependencies

```txt
# requirements.txt - Good
torch==2.1.0
transformers==4.30.2
numpy==1.24.3

# Bad - versions can change and break builds
torch
transformers
numpy
```

### 2. Use the Recommended Base Image

```python
# Recommended
.from_base("parachutes/python:3.12")

# Not recommended unless you know what you're doing
.from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
```

### 3. Optimize Layer Order

Put things that change less frequently earlier in your Image definition:

1. System packages
2. Python packages (requirements.txt)
3. Application code

### 4. Clean Up in Commands

```python
# Good: Clean up in the same layer
.run_command("""
    apt-get update &&
    apt-get install -y git curl &&
    rm -rf /var/lib/apt/lists/*
""")

# Less optimal: Separate commands create more layers
.run_command("apt-get update")
.run_command("apt-get install -y git curl")
```

### 5. Review Build Context

Always review which files will be uploaded before confirming:

```bash
Found 15 files to include in build context
 requirements.txt
 src/main.py
 ...
Confirm submitting build context? (y/n)
```

Make sure no sensitive files (`.env`, credentials) are included.

## Next Steps

- **[Deploying Chutes](/docs/cli/deploy)** - Deploy your built images
- **[Managing Resources](/docs/cli/manage)** - Manage your chutes and images
- **[Account Management](/docs/cli/account)** - API keys and configuration
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
