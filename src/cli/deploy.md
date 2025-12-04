# Deploying Chutes

The `chutes deploy` command takes your built images and deploys them as live, scalable AI applications on the Chutes platform.

## Basic Deploy Command

### `chutes deploy`

Deploy a chute to the platform.

```bash
chutes deploy <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image for the chute
- `--debug`: Enable debug logging
- `--public`: Mark chute as public/available to anyone
- `--accept-fee`: Acknowledge and accept the deployment fee

## Deployment Examples

### Basic Deployment

```bash
# Deploy with fee acknowledgment
chutes deploy my_chute:chute --accept-fee
```

**What happens:**

- ✅ Validates image exists and is built
- ✅ Creates deployment configuration
- ✅ Registers chute with the platform
- ✅ Returns chute ID and version

### Production Deployment

```bash
# Deploy with logo
chutes deploy my_chute:chute \
  --logo ./assets/logo.png \
  --accept-fee
```

### Private vs Public Deployments

```bash
# Private deployment (default) - only you can access
chutes deploy my_chute:chute --accept-fee

# Public deployment (requires special permissions)
chutes deploy my_chute:chute --public --accept-fee
```

> **Note:** Public chutes require special permissions. If you need to share your chute, use the `chutes share` command instead.

## Deployment Process

### Deployment Stages

```bash
# Example deployment output
Deploying chute: my_chute:chute
You are about to upload my_chute.py and deploy my-chute, confirm? (y/n) y
Successfully deployed chute my-chute chute_id=abc123 version=1
```

### What Gets Deployed

When you deploy, the following is sent to the platform:

- **Chute Configuration**: Name, readme, tagline
- **Node Selector**: GPU requirements
- **Cords**: API endpoints your chute exposes
- **Code Reference**: Your chute's Python code
- **Image Reference**: The built image to use

## Deployment Fees

Deployment incurs a one-time fee based on your NodeSelector configuration:

```bash
# Deploy and acknowledge the fee
chutes deploy my_chute:chute --accept-fee
```

If you don't include `--accept-fee`, you may receive a 402 error indicating the deployment fee needs to be acknowledged.

### Fee Structure

Deployment fees are calculated based on:
- **GPU Type**: Higher-end GPUs cost more
- **GPU Count**: More GPUs = higher fee
- **VRAM Requirements**: Higher VRAM requirements cost more

Example fee calculation:
- Single RTX 3090 at $0.12/hr = $0.36 deployment fee
- Multiple GPUs or premium GPUs will have higher fees

## Pre-Deployment Checklist

Before deploying, ensure:

### 1. Image is Built and Ready

```bash
# Check image status
chutes images list --name my-image
chutes images get my-image

# Should show status: "built and pushed"
```

### 2. Chute Configuration is Correct

```python
# Verify your chute definition
from chutes.chute import Chute, NodeSelector

chute = Chute(
    username="myuser",
    name="my-chute",
    tagline="My awesome AI chute",
    readme="## My Chute\n\nDescription here...",
    image=my_image,
    concurrency=4,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
    ),
)
```

### 3. Cords are Defined

```python
@chute.cord()
async def my_function(self, input_data: str) -> str:
    return f"Processed: {input_data}"

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
)
async def generate(self, prompt: str) -> str:
    # Your logic here
    return result
```

## Chute Configuration Options

### NodeSelector

Control which GPUs your chute runs on:

```python
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count=1,              # Number of GPUs (1-8)
    min_vram_gb_per_gpu=16,   # Minimum VRAM per GPU (16-80)
    include=["rtx4090"],      # Only use these GPU types
    exclude=["rtx3090"],      # Don't use these GPU types
)
```

### Concurrency

Set how many concurrent requests your chute can handle:

```python
chute = Chute(
    ...
    concurrency=4,  # Handle 4 concurrent requests per instance
)
```

### Auto-Scaling

Configure automatic scaling behavior:

```python
chute = Chute(
    ...
    max_instances=10,           # Maximum number of instances
    scaling_threshold=0.8,      # Scale up threshold
    shutdown_after_seconds=300, # Shutdown idle instances after 5 minutes
)
```

### Network Egress

Control external network access:

```python
chute = Chute(
    ...
    allow_external_egress=True,  # Allow external network access
)
```

> **Note:** By default, `allow_external_egress` is **true** for custom chutes but **false** for vllm/sglang templates. Set to `True` if your chute needs to fetch external resources (e.g., image URLs for vision models).

## Sharing Chutes

After deployment, you can share your chute with other users:

```bash
# Share with another user
chutes share --chute-id my-chute --user-id colleague

# Remove sharing
chutes share --chute-id my-chute --user-id colleague --remove
```

### Billing When Sharing

When you share a chute:
- **You** (chute owner) pay the hourly rate while instances are running
- **The user you shared with** pays the standard usage rate (per token, per step, etc.)

## Troubleshooting Deployments

### Common Deployment Issues

**"Image is not available to be used (yet)!"**

```bash
# Image hasn't finished building - check status
chutes images get my-image

# Wait for status: "built and pushed"
```

**"Unable to create public chutes from non-public images"**

```bash
# If deploying public chute, image must also be public
# Rebuild image with --public flag
chutes build my_chute:chute --public --wait
```

**402 Payment Required**

```bash
# Include --accept-fee flag
chutes deploy my_chute:chute --accept-fee
```

**409 Conflict**

```bash
# Chute with this name already exists
# Delete existing chute first
chutes chutes delete my-chute

# Or use a different name in your chute definition
```

### Debug Commands

```bash
# Enable debug logging
chutes deploy my_chute:chute --debug --accept-fee

# Check existing chutes
chutes chutes list
chutes chutes get my-chute

# Check image status
chutes images get my-image
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Deploy to Chutes
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Chutes
        run: pip install chutes

      - name: Configure Chutes
        env:
          CHUTES_CONFIG: ${{ secrets.CHUTES_CONFIG }}
        run: |
          mkdir -p ~/.chutes
          echo "$CHUTES_CONFIG" > ~/.chutes/config.ini

      - name: Build and Deploy
        run: |
          chutes build my_app:chute --wait
          chutes deploy my_app:chute --accept-fee
```

### GitLab CI

```yaml
deploy:
  stage: deploy
  script:
    - pip install chutes
    - mkdir -p ~/.chutes
    - echo "$CHUTES_CONFIG" > ~/.chutes/config.ini
    - chutes build my_app:chute --wait
    - chutes deploy my_app:chute --accept-fee
  only:
    - main
```

## Production Deployment Checklist

### Pre-Deployment

```bash
# ✅ Run tests locally
python -m pytest tests/

# ✅ Build image and verify
chutes build my_chute:chute --wait
chutes images get my-chute

# ✅ Test locally if possible
docker run --rm -it -p 8000:8000 my_chute:tag chutes run my_chute:chute --dev
```

### Deployment

```bash
# ✅ Deploy with fee acknowledgment
chutes deploy my_chute:chute --accept-fee

# ✅ Note the chute_id and version from output
```

### Post-Deployment

```bash
# ✅ Verify deployment
chutes chutes get my-chute

# ✅ Warm up the chute
chutes warmup my-chute

# ✅ Test the endpoint
curl -X POST https://your-chute-url/your-endpoint \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}'
```

## Best Practices

### 1. Use Meaningful Names

```python
chute = Chute(
    name="sentiment-analyzer-v2",  # Clear, versioned name
    tagline="Analyze sentiment in text using BERT",
    readme="## Sentiment Analyzer\n\n...",
)
```

### 2. Set Appropriate Concurrency

```python
# For LLMs with continuous batching (vllm/sglang)
concurrency=64

# For single-request models (diffusion, custom)
concurrency=1

# For models with some parallelism
concurrency=4
```

### 3. Configure Shutdown Timer

```python
# For development/testing - short timeout
shutdown_after_seconds=60

# For production - longer timeout to avoid cold starts
shutdown_after_seconds=300
```

### 4. Right-Size GPU Requirements

```python
# Match your model's actual requirements
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,  # For ~13B parameter models
)

# Don't over-provision
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,  # Only if you actually need A100
)
```

## Next Steps

- **[Managing Resources](/docs/cli/manage)** - Monitor and manage deployments
- **[Building Images](/docs/cli/build)** - Optimize your build process
- **[Account Management](/docs/cli/account)** - API keys and configuration
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
