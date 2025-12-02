# CLI Command Overview

The Chutes CLI provides a complete set of commands for managing your AI applications, from account setup to deployment and monitoring.

## Installation

The CLI is included when you install the Chutes SDK:

```bash
pip install chutes
```

Verify installation:

```bash
chutes --help
```

## Command Structure

All Chutes commands follow this pattern:

```bash
chutes <command> [subcommand] [options] [arguments]
```

## Account Management

### `chutes register`

Create a new account with the Chutes platform.

```bash
chutes register [OPTIONS]
```

**Options:**

- `--config-path TEXT`: Custom path to config file
- `--username TEXT`: Desired username
- `--wallets-path TEXT`: Path to Bittensor wallets directory (default: `~/.bittensor/wallets`)
- `--wallet TEXT`: Name of the wallet to use
- `--hotkey TEXT`: Hotkey to register with

**Example:**

```bash
chutes register --username myuser
```

## Building & Deployment

### `chutes build`

Build a Docker image for your chute.

```bash
chutes build <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image
- `--local`: Build locally instead of remotely
- `--debug`: Enable debug logging
- `--include-cwd`: Include entire current directory in build context
- `--wait`: Wait for build to complete
- `--public`: Mark image as public

**Examples:**

```bash
# Build remotely and wait for completion
chutes build my_chute:chute --wait

# Build locally for testing
chutes build my_chute:chute --local

# Build with a logo and make public
chutes build my_chute:chute --logo ./logo.png --public
```

### `chutes deploy`

Deploy a chute to the platform.

```bash
chutes deploy <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image
- `--debug`: Enable debug logging
- `--public`: Mark chute as public
- `--accept-fee`: Acknowledge the deployment fee and accept being charged

**Examples:**

```bash
# Basic deployment
chutes deploy my_chute:chute

# Deploy with logo
chutes deploy my_chute:chute --logo ./logo.png

# Deploy and accept the deployment fee
chutes deploy my_chute:chute --accept-fee
```

### `chutes run`

Run a chute locally for development and testing.

```bash
chutes run <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--host TEXT`: Host to bind to (default: 0.0.0.0)
- `--port INTEGER`: Port to listen on (default: 8000)
- `--debug`: Enable debug logging
- `--dev`: Enable development mode

**Examples:**

```bash
# Run on default port
chutes run my_chute:chute --dev

# Run on custom port with debug
chutes run my_chute:chute --port 8080 --debug --dev
```

### `chutes share`

Share a chute with another user.

```bash
chutes share [OPTIONS]
```

**Options:**

- `--chute-id TEXT`: The chute UUID or name to share (required)
- `--user-id TEXT`: The user UUID or username to share with (required)
- `--config-path TEXT`: Custom config path
- `--remove`: Unshare/remove the share instead of adding

**Examples:**

```bash
# Share a chute with another user
chutes share --chute-id my-chute --user-id anotheruser

# Remove sharing
chutes share --chute-id my-chute --user-id anotheruser --remove
```

### `chutes warmup`

Warm up a chute to ensure an instance is ready for requests.

```bash
chutes warmup <chute_id_or_ref> [OPTIONS]
```

**Arguments:**

- `chute_id_or_ref`: The chute UUID, name, or file reference (format: `filename:chutevarname`)

**Options:**

- `--config-path TEXT`: Custom config path
- `--debug`: Enable debug logging

**Example:**

```bash
chutes warmup my-chute
```

## Resource Management

### `chutes chutes`

Manage your deployed chutes.

#### `chutes chutes list`

List your chutes.

```bash
chutes chutes list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)
- `--include-public`: Include public chutes

**Example:**

```bash
chutes chutes list --limit 10 --include-public
```

#### `chutes chutes get`

Get detailed information about a specific chute.

```bash
chutes chutes get <name_or_id>
```

**Example:**

```bash
chutes chutes get my-awesome-chute
```

#### `chutes chutes delete`

Delete a chute.

```bash
chutes chutes delete <name_or_id>
```

**Example:**

```bash
chutes chutes delete my-old-chute
```

### `chutes images`

Manage your Docker images.

#### `chutes images list`

List your images.

```bash
chutes images list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)
- `--include-public`: Include public images

#### `chutes images get`

Get detailed information about a specific image.

```bash
chutes images get <name_or_id>
```

#### `chutes images delete`

Delete an image.

```bash
chutes images delete <name_or_id>
```

### `chutes keys`

Manage API keys.

#### `chutes keys create`

Create a new API key.

```bash
chutes keys create [OPTIONS]
```

**Options:**

- `--name TEXT`: Name for the API key (required)
- `--admin`: Create admin key with full permissions
- `--images`: Allow full access to images
- `--chutes`: Allow full access to chutes
- `--image-ids TEXT`: Specific image IDs to allow (can be repeated)
- `--chute-ids TEXT`: Specific chute IDs to allow (can be repeated)
- `--action [read|write|delete|invoke]`: Specify action scope
- `--json-input TEXT`: Provide raw scopes document as JSON for advanced usage
- `--config-path TEXT`: Custom config path

**Examples:**

```bash
# Admin key
chutes keys create --name admin-key --admin

# Key with invoke access to all chutes
chutes keys create --name invoke-key --chutes --action invoke

# Key with access to specific chute
chutes keys create --name readonly-key --chute-ids 12345 --action read
```

#### `chutes keys list`

List your API keys.

```bash
chutes keys list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)

#### `chutes keys get`

Get details about a specific API key.

```bash
chutes keys get <name_or_id>
```

#### `chutes keys delete`

Delete an API key.

```bash
chutes keys delete <name_or_id>
```

### `chutes secrets`

Manage secrets for your chutes (e.g., HuggingFace tokens for private models).

#### `chutes secrets create`

Create a new secret.

```bash
chutes secrets create [OPTIONS]
```

**Options:**

- `--purpose TEXT`: The chute UUID or name this secret is for (required)
- `--key TEXT`: The secret key/name (required)
- `--value TEXT`: The secret value (required)
- `--config-path TEXT`: Custom config path

**Example:**

```bash
chutes secrets create --purpose my-chute --key HF_TOKEN --value hf_xxxxxxxxxxxx
```

#### `chutes secrets list`

List your secrets.

```bash
chutes secrets list [OPTIONS]
```

**Options:**

- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)

#### `chutes secrets get`

Get details about a specific secret.

```bash
chutes secrets get <secret_id>
```

#### `chutes secrets delete`

Delete a secret.

```bash
chutes secrets delete <secret_id>
```

## Utilities

### `chutes report`

Report an invocation for billing/tracking purposes.

```bash
chutes report [OPTIONS]
```

### `chutes refinger`

Change your fingerprint.

```bash
chutes refinger [OPTIONS]
```

## Global Options

These options work with most commands:

- `--help`: Show help message
- `--config-path TEXT`: Path to custom config file
- `--debug`: Enable debug logging

## Configuration

### Config File Location

Default: `~/.chutes/config.ini`

Override with:

```bash
export CHUTES_CONFIG_PATH=/path/to/config.ini
```

### Environment Variables

- `CHUTES_CONFIG_PATH`: Custom config file path
- `CHUTES_API_URL`: API base URL
- `CHUTES_ALLOW_MISSING`: Allow missing config

## Common Workflows

### 1. First-Time Setup

```bash
# Register account
chutes register

# Create admin API key
chutes keys create --name admin --admin
```

### 2. Develop and Deploy

```bash
# Build your image
chutes build my_app:chute --wait

# Test locally
docker run --rm -it -e CHUTES_EXECUTION_CONTEXT=REMOTE -p 8000:8000 my_app:tag chutes run my_app:chute --port 8000 --dev

# Deploy to production
chutes deploy my_app:chute --accept-fee
```

### 3. Manage Resources

```bash
# List your chutes
chutes chutes list

# Get detailed info
chutes chutes get my-app

# Warm up a chute
chutes warmup my-app

# Share with another user
chutes share --chute-id my-app --user-id colleague

# Clean up old resources
chutes chutes delete old-chute
chutes images delete old-image
```

## Troubleshooting

### Common Issues

**Command not found**

```bash
# Check installation
pip show chutes

# Try with Python module
python -m chutes --help
```

**Authentication errors**

```bash
# Re-register if needed
chutes register

# Check config file
cat ~/.chutes/config.ini
```

**Build failures**

```bash
# Try local build for debugging
chutes build my_app:chute --local --debug

# Check image syntax
python -c "from my_app import chute; print(chute.image)"
```

**Deployment issues**

```bash
# Verify image exists and is built
chutes images list --name my-image
chutes images get my-image

# Check chute status
chutes chutes get my-chute
```

### Debug Mode

Enable debug logging for detailed output:

```bash
chutes build my_app:chute --debug
```

## Getting Help

### Built-in Help

```bash
# General help
chutes --help

# Command-specific help
chutes build --help
chutes deploy --help
chutes chutes list --help
```

### Support Resources

- 📖 **Documentation**: [Complete Docs](/docs)
- 💬 **Discord**: [Community Chat](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/chutesai/chutes/issues)

---

Continue to specific command documentation:

- **[Account Management](/docs/cli/account)** - Detailed account commands
- **[Building Images](/docs/cli/build)** - Advanced build options
- **[Deploying Chutes](/docs/cli/deploy)** - Deployment strategies
- **[Managing Resources](/docs/cli/manage)** - Resource management
