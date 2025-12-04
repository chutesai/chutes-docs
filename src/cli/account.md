# Account Management

This section covers CLI commands for managing your Chutes account, registration, authentication, and API keys.

## Account Registration

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

**Examples:**

```bash
# Basic registration with interactive prompts
chutes register

# Register with specific username
chutes register --username myusername

# Register with specific wallet
chutes register --wallet my_wallet --hotkey my_hotkey
```

**Registration Process:**

1. **Choose Username**: Select a unique username for your account
2. **Wallet Selection**: Choose from available Bittensor wallets
3. **Hotkey Selection**: Select which hotkey to use for signing
4. **Token Verification**: Complete registration token verification
5. **Config Generation**: Configuration file is generated and saved

**What Happens During Registration:**

- Creates your Chutes account
- Generates initial configuration file at `~/.chutes/config.ini`
- Sets up your payment address for adding balance
- Provides your account fingerprint (keep this safe!)

## API Key Management

API keys provide programmatic access to your Chutes account and are essential for CI/CD and automation.

### `chutes keys list`

List all API keys for your account.

```bash
chutes keys list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)

**Example:**

```bash
chutes keys list
```

**Output:**

```
┌──────────┬─────────────────────┬─────────┬──────────────────────────┐
│ ID       │ Name                │ Admin   │ Scopes                   │
├──────────┼─────────────────────┼─────────┼──────────────────────────┤
│ key_123  │ admin               │ true    │ -                        │
│ key_456  │ ci-cd               │ false   │ {"action": "invoke"...}  │
│ key_789  │ dev                 │ false   │ {"action": "read"...}    │
└──────────┴─────────────────────┴─────────┴──────────────────────────┘
```

### `chutes keys create`

Create a new API key.

```bash
chutes keys create [OPTIONS]
```

**Options:**

- `--name TEXT`: Name for the API key (required)
- `--admin`: Create admin key with full permissions
- `--images`: Allow full access to images
- `--chutes`: Allow full access to chutes
- `--image-ids TEXT`: Allow access to specific image IDs (can be repeated)
- `--chute-ids TEXT`: Allow access to specific chute IDs (can be repeated)
- `--action [read|write|delete|invoke]`: Specify action scope
- `--json-input TEXT`: Provide raw scopes document as JSON for advanced usage
- `--config-path TEXT`: Custom config path

**Examples:**

```bash
# Create admin key with full permissions
chutes keys create --name admin --admin

# Create key for invoking all chutes
chutes keys create --name invoke-all --chutes --action invoke

# Create key for reading specific chute
chutes keys create --name readonly-key --chute-ids my-chute-id --action read

# Create key for managing images
chutes keys create --name image-manager --images --action write

# Create key with advanced scopes using JSON
chutes keys create --name advanced-key --json-input '{"scopes": [{"object_type": "chutes", "action": "invoke"}]}'
```

**Key Types:**

- **Admin Keys**: Full account access including all resources
- **Scoped Keys**: Limited access based on object type and action

**Using Your API Key:**

After creating a key, you'll receive output like:

```
API key created successfully
{
  "api_key_id": "...",
  "name": "my-key",
  "secret_key": "cpk_xxxxxxxxxxxxxxxx"
}

To use the key, add "Authorization: Basic cpk_xxxxxxxxxxxxxxxx" to your headers!
```

### `chutes keys get`

Get details about a specific API key.

```bash
chutes keys get <name_or_id>
```

**Example:**

```bash
chutes keys get my-key
```

### `chutes keys delete`

Delete an API key.

```bash
chutes keys delete <name_or_id>
```

**Example:**

```bash
# Delete by name
chutes keys delete old-key
```

**Safety Notes:**

- Deleted keys cannot be recovered
- Active deployments using the key will lose access
- Always rotate keys before deletion in production

## Secrets Management

Secrets allow you to securely store sensitive values (like API tokens) that your chutes need at runtime.

### `chutes secrets create`

Create a new secret for a chute.

```bash
chutes secrets create [OPTIONS]
```

**Options:**

- `--purpose TEXT`: The chute UUID or name this secret is for (required)
- `--key TEXT`: The secret key/environment variable name (required)
- `--value TEXT`: The secret value (required)
- `--config-path TEXT`: Custom config path

**Examples:**

```bash
# Create a HuggingFace token secret for a chute
chutes secrets create --purpose my-llm-chute --key HF_TOKEN --value hf_xxxxxxxxxxxx

# Create an API key secret
chutes secrets create --purpose my-chute --key EXTERNAL_API_KEY --value sk-xxxxxxxx
```

### `chutes secrets list`

List your secrets.

```bash
chutes secrets list [OPTIONS]
```

**Options:**

- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)

**Output:**

```
┌────────────────┬─────────────────┬─────────────┬─────────────────────┐
│ Secret ID      │ Purpose         │ Key         │ Created             │
├────────────────┼─────────────────┼─────────────┼─────────────────────┤
│ sec_123abc     │ my-llm-chute    │ HF_TOKEN    │ 2024-01-15 10:30:00 │
│ sec_456def     │ my-chute        │ API_KEY     │ 2024-01-20 14:45:00 │
└────────────────┴─────────────────┴─────────────┴─────────────────────┘
```

### `chutes secrets get`

Get details about a specific secret.

```bash
chutes secrets get <secret_id>
```

### `chutes secrets delete`

Delete a secret.

```bash
chutes secrets delete <secret_id>
```

## Configuration Management

### Config File Structure

The Chutes configuration file (`~/.chutes/config.ini`) stores your account settings:

```ini
[api]
base_url = https://api.chutes.ai

[auth]
username = myusername
user_id = user_123abc456def
hotkey_seed = your_hotkey_seed
hotkey_name = my_hotkey
hotkey_ss58address = 5xxxxx...

[payment]
address = 5xxxxx...
```

### Environment Variables

Override config settings with environment variables:

```bash
# Config path
export CHUTES_CONFIG_PATH=/path/to/config.ini

# API URL (for development/testing)
export CHUTES_API_URL=https://api.chutes.ai

# Allow missing config (useful during registration)
export CHUTES_ALLOW_MISSING=true
```

### Multiple Configurations

Manage multiple accounts or environments:

```bash
# Create environment-specific configs
mkdir -p ~/.chutes/environments

# Production config
chutes register --config-path ~/.chutes/environments/prod.ini

# Staging config
chutes register --config-path ~/.chutes/environments/staging.ini

# Use specific config for commands
chutes build my_app:chute --config-path ~/.chutes/environments/prod.ini
```

## Security Best Practices

### API Key Security

```bash
# Use separate keys for different purposes
chutes keys create --name production-deploy --chutes --action write
chutes keys create --name monitoring --chutes --action read
chutes keys create --name ci-invoke --chutes --action invoke

# Rotate keys regularly
chutes keys create --name new-prod-key --admin
# Update your deployments to use new key
chutes keys delete old-prod-key
```

### Account Security

- **Keep Your Fingerprint Safe**: Your fingerprint is shown during registration - don't share it
- **Secure Your Hotkey**: The hotkey seed in your config file should be kept private
- **Regular Audits**: Review your API keys periodically and delete unused ones
- **Environment Separation**: Use different keys for dev/staging/prod

### CI/CD Security

```yaml
# GitHub Actions example
env:
  CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }}

steps:
  - name: Deploy to Chutes
    run: |
      pip install chutes
      mkdir -p ~/.chutes
      cat > ~/.chutes/config.ini << EOF
      [api]
      base_url = https://api.chutes.ai
      
      [auth]
      # Use API key authentication
      EOF
      chutes deploy my_app:chute --accept-fee
```

## Troubleshooting

### Common Issues

**Registration fails?**

```bash
# Check network connectivity
curl -I https://api.chutes.ai/ping

# Try with different username (may already be taken)
chutes register --username alternative_username

# Verify wallet path exists
ls ~/.bittensor/wallets/
```

**API key not working?**

```bash
# Verify key exists and check scopes
chutes keys list
chutes keys get my-key

# Ensure you're using the secret_key value with "Authorization: Basic" header
```

**Configuration issues?**

```bash
# Check config file exists and has correct format
cat ~/.chutes/config.ini

# Verify environment variables aren't overriding
echo $CHUTES_CONFIG_PATH
echo $CHUTES_API_URL
```

### Getting Help

- **Account Issues**: [Discord Community](https://discord.gg/wHrXwWkCRz)
- **Technical Support**: [GitHub Issues](https://github.com/chutesai/chutes/issues)
- **Documentation**: [Chutes Docs](https://chutes.ai/docs)

## Next Steps

- **[Building Images](/docs/cli/build)** - Learn to build Docker images
- **[Deploying Chutes](/docs/cli/deploy)** - Deploy your applications
- **[Managing Resources](/docs/cli/manage)** - Manage your deployments
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
