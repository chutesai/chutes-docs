# Managing Resources

This section covers CLI commands for managing your deployed chutes, images, API keys, and secrets.

## Chute Management

### `chutes chutes list`

List all your deployed chutes.

```bash
chutes chutes list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)
- `--include-public`: Include public chutes in results

**Examples:**

```bash
# List all your chutes
chutes chutes list

# Filter by name
chutes chutes list --name sentiment

# Include public chutes
chutes chutes list --include-public --limit 50
```

**Output:**

```
┌─────────────────┬─────────────────────┬────────┬───────────────────────────────┐
│ ID              │ Name                │ Status │ Cords                         │
├─────────────────┼─────────────────────┼────────┼───────────────────────────────┤
│ chute_abc123    │ sentiment-api       │ hot    │ analyze                       │
│                 │                     │        │   stream=False                │
│                 │                     │        │   POST /analyze               │
├─────────────────┼─────────────────────┼────────┼───────────────────────────────┤
│ chute_def456    │ image-gen           │ cold   │ generate                      │
│                 │                     │        │   stream=True                 │
│                 │                     │        │   POST /generate              │
└─────────────────┴─────────────────────┴────────┴───────────────────────────────┘
```

### `chutes chutes get`

Get detailed information about a specific chute.

```bash
chutes chutes get <name_or_id>
```

**Arguments:**

- `name_or_id`: Name or UUID of the chute

**Example:**

```bash
chutes chutes get my-chute
```

**Output:**

```json
{
  "chute_id": "abc123-def456-...",
  "name": "my-chute",
  "tagline": "My awesome AI chute",
  "slug": "myuser/my-chute",
  "hot": true,
  "created_at": "2024-01-15T10:30:00Z",
  "node_selector": {
    "gpu_count": 1,
    "min_vram_gb_per_gpu": 24
  },
  ...
}
```

### `chutes chutes delete`

Delete a chute and all its resources.

```bash
chutes chutes delete <name_or_id>
```

**Arguments:**

- `name_or_id`: Name or UUID of the chute to delete

**Example:**

```bash
chutes chutes delete old-chute
```

**Confirmation:**

```
Are you sure you want to delete chutes/old-chute? This action is irreversible. (y/n): y
Successfully deleted chute chute_abc123
```

> **⚠️ Warning:** Deletion is permanent and cannot be undone!

## Image Management

### `chutes images list`

List all your Docker images.

```bash
chutes images list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)
- `--include-public`: Include public images in results

**Examples:**

```bash
# List all your images
chutes images list

# Filter by name
chutes images list --name my-app

# Include public images
chutes images list --include-public
```

**Output:**

```
┌─────────────────┬─────────────────┬─────────┬──────────────────┬─────────────────────┐
│ ID              │ Name            │ Tag     │ Status           │ Created             │
├─────────────────┼─────────────────┼─────────┼──────────────────┼─────────────────────┤
│ img_abc123      │ sentiment-api   │ 1.0     │ built and pushed │ 2024-01-15 10:30:00 │
│ img_def456      │ image-gen       │ 2.1     │ built and pushed │ 2024-01-20 14:45:00 │
│ img_ghi789      │ test-app        │ dev     │ building         │ 2024-01-25 09:15:00 │
└─────────────────┴─────────────────┴─────────┴──────────────────┴─────────────────────┘
```

### `chutes images get`

Get detailed information about a specific image.

```bash
chutes images get <name_or_id>
```

**Arguments:**

- `name_or_id`: Name or UUID of the image

**Example:**

```bash
chutes images get my-app
```

### `chutes images delete`

Delete an image.

```bash
chutes images delete <name_or_id>
```

**Arguments:**

- `name_or_id`: Name or UUID of the image to delete

**Example:**

```bash
chutes images delete old-image:1.0
```

> **Note:** You cannot delete images that are currently in use by deployed chutes.

## Sharing Chutes

### `chutes share`

Share a chute with another user or remove sharing.

```bash
chutes share [OPTIONS]
```

**Options:**

- `--chute-id TEXT`: The chute UUID or name to share (required)
- `--user-id TEXT`: The user UUID or username to share with (required)
- `--config-path TEXT`: Custom config path
- `--remove`: Remove sharing instead of adding

**Examples:**

```bash
# Share a chute with another user
chutes share --chute-id my-chute --user-id colleague

# Share by UUIDs
chutes share --chute-id abc123-def456 --user-id user789-xyz

# Remove sharing
chutes share --chute-id my-chute --user-id colleague --remove
```

### Sharing and Billing

When you share a chute:

- **Chute Owner**: Pays the hourly compute rate while instances are running
- **Shared User**: Pays the standard invocation rate (per token, per step, etc.)

This allows you to provide access to your deployed models while sharing the costs appropriately.

## Warming Up Chutes

### `chutes warmup`

Warm up a chute to ensure an instance is ready to handle requests.

```bash
chutes warmup <chute_id_or_ref> [OPTIONS]
```

**Arguments:**

- `chute_id_or_ref`: The chute UUID, name, or file reference (`filename:chutevarname`)

**Options:**

- `--config-path TEXT`: Custom config path
- `--debug`: Enable debug logging

**Examples:**

```bash
# Warm up by name
chutes warmup my-chute

# Warm up by UUID
chutes warmup abc123-def456

# Warm up from file reference
chutes warmup my_chute:chute
```

**Output:**

```
Status: cold -- Starting instance...
Status: warming -- Loading model...
Status: hot -- Instance is ready!
```

Use warmup to reduce latency for the first request to a cold chute.

## Common Workflows

### Deploying Updates

```bash
# 1. Build new image
chutes build my_chute:chute --wait

# 2. Delete old chute (if needed)
chutes chutes delete my-chute

# 3. Deploy new version
chutes deploy my_chute:chute --accept-fee

# 4. Warm up
chutes warmup my-chute
```

### Cleaning Up Resources

**Important:** You must delete chutes *before* deleting the images they use. Images tied to existing chutes (even if not currently running) cannot be deleted.

```bash
# List all chutes
chutes chutes list

# Delete unused chutes first
chutes chutes delete old-chute-1
chutes chutes delete old-chute-2

# List all images
chutes images list

# Delete unused images (after their chutes are removed)
chutes images delete old-image:1.0
chutes images delete test-image:dev
```

### Sharing with Team Members

```bash
# Share with multiple users
chutes share --chute-id my-model --user-id alice
chutes share --chute-id my-model --user-id bob
chutes share --chute-id my-model --user-id charlie

# Later, remove access
chutes share --chute-id my-model --user-id bob --remove
```

## Automation and Scripting

### Bash Scripting

```bash
#!/bin/bash

# Deploy and warm up script
set -e

CHUTE_REF="my_chute:chute"
CHUTE_NAME="my-chute"

echo "Building image..."
chutes build $CHUTE_REF --wait

echo "Deploying chute..."
chutes deploy $CHUTE_REF --accept-fee

echo "Warming up..."
chutes warmup $CHUTE_NAME

echo "Deployment complete!"
```

### Python Scripting

```python
#!/usr/bin/env python3
import subprocess
import sys

def run_command(command):
    """Run a chutes CLI command."""
        result = subprocess.run(
            f"chutes {command}".split(),
            capture_output=True,
        text=True
        )
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    # List all chutes
    print("Your chutes:")
    output = run_command("chutes list")
    print(output)
    
    # Check specific chute
    print("\nChute details:")
    output = run_command("chutes get my-chute")
    print(output)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

**Chute not found?**

```bash
# Check exact name/ID
chutes chutes list

# Use the exact name or UUID from the list
chutes chutes get exact-chute-name
```

**Cannot delete chute?**

The deletion requires confirmation. Type `y` when prompted:

```bash
chutes chutes delete my-chute
# Are you sure you want to delete chutes/my-chute? This action is irreversible. (y/n): y
```

**Image status not "built and pushed"?**

```bash
# Check image status
chutes images get my-image

# If status is "building", wait for build to complete
# If status shows an error, rebuild the image
chutes build my_chute:chute --wait
```

**Warmup fails?**

```bash
# Enable debug logging
chutes warmup my-chute --debug

# Check chute exists
chutes chutes get my-chute
```

## Best Practices

### 1. Regular Cleanup

Periodically review and delete unused resources:

```bash
# Review chutes
chutes chutes list

# Review images
chutes images list

# Delete what you no longer need
chutes chutes delete unused-chute
chutes images delete old-image:tag
```

### 2. Use Descriptive Names

Name your chutes and images clearly:

```
# Good
sentiment-analyzer-bert-v2
image-gen-sdxl-1.0
llm-llama3-8b-instruct

# Not as good
test1
my-app
chute
```

### 3. Warm Up Before Critical Usage

If you need low latency, warm up your chute before sending requests:

```bash
chutes warmup my-chute
# Wait for "hot" status
# Then send your requests
```

### 4. Share Instead of Making Public

For most use cases, sharing with specific users is better than making chutes public:

```bash
# Better: Share with specific users
chutes share --chute-id my-chute --user-id trusted-user

# Only if needed: Deploy as public (requires permissions)
chutes deploy my_chute:chute --public --accept-fee
```

## Next Steps

- **[Building Images](/docs/cli/build)** - Optimize your images
- **[Deploying Chutes](/docs/cli/deploy)** - Advanced deployment strategies
- **[Account Management](/docs/cli/account)** - API keys and billing
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
