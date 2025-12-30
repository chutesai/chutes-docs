# Troubleshooting the CLI

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
- 📨 **Support**: [Email](support@chutes.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/chutesai/chutes/issues)

---

Continue to specific command documentation:

- **[Account Management](/docs/cli/account)** - Detailed account commands
- **[Building Images](/docs/cli/build)** - Advanced build options
- **[Deploying Chutes](/docs/cli/deploy)** - Deployment strategies
- **[Managing Resources](/docs/cli/manage)** - Resource management
