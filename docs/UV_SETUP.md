# UV Package Manager Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## Why UV?

- **10-100x faster** than pip for package installation
- **Reliable** - uses lock files for reproducible builds
- **Drop-in replacement** for pip - same commands
- **Built in Rust** - extremely fast dependency resolution
- **Works with pyproject.toml** - modern Python packaging

## Installation

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Using Make

```bash
make install-uv
```

### Verify Installation

```bash
uv --version
```

You should see something like: `uv 0.x.x`

## Usage in This Project

### Create Virtual Environment

```bash
# UV automatically uses Python 3.13 if available
uv venv

# Or specify Python version explicitly
uv venv --python 3.13
```

### Activate Virtual Environment

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Install Dependencies

```bash
# Install all dependencies (including dev)
uv pip install -e ".[dev]"

# Install production dependencies only
uv pip install -e .

# Add a new package
uv pip install package-name

# Add to pyproject.toml and install
uv pip install -e ".[dev]"
```

### Sync Dependencies

```bash
# Install exact versions from lock file (when implemented)
uv pip sync
```

## Common Commands

| Task | UV Command | Old (pip) |
|------|------------|-----------|
| Install deps | `uv pip install -e ".[dev]"` | `pip install -e ".[dev]"` |
| Add package | `uv pip install requests` | `pip install requests` |
| Remove package | `uv pip uninstall requests` | `pip uninstall requests` |
| List packages | `uv pip list` | `pip list` |
| Freeze deps | `uv pip freeze` | `pip freeze` |

## UV vs Pip Performance

Typical installation times for this project:

| Package Manager | Time | Speedup |
|----------------|------|---------|
| **uv** | ~10 seconds | **1x** |
| pip | ~2-5 minutes | 12-30x slower |
| poetry | ~3-6 minutes | 18-36x slower |

## Project Setup with UV

### Full Automated Setup

```bash
# This checks for UV, creates venv, and installs everything
make setup
```

### Manual Setup

```bash
# 1. Install uv
make install-uv

# 2. Create venv
uv venv --python 3.13

# 3. Activate
source .venv/bin/activate

# 4. Install dependencies
uv pip install -e ".[dev]"

# 5. Copy .env
cp .env.example .env
```

## Updating Dependencies

### Update a Single Package

```bash
uv pip install --upgrade package-name
```

### Update All Packages

```bash
uv pip install --upgrade -e ".[dev]"
```

### Check for Outdated Packages

```bash
uv pip list --outdated
```

## Troubleshooting

### UV Not Found After Installation

**Problem:** `command not found: uv`

**Solution:**
```bash
# Restart shell or reload PATH
source ~/.cargo/env

# Or add to your shell profile (~/.bashrc, ~/.zshrc):
export PATH="$HOME/.cargo/bin:$PATH"
```

### Virtual Environment Issues

**Problem:** Dependencies not found after installation

**Solution:**
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall
uv pip install -e ".[dev]"
```

### Permission Errors

**Problem:** Permission denied when installing

**Solution:**
```bash
# Don't use sudo with uv/pip - use virtual environments
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Advanced UV Features

### Lock Files (Future)

UV supports lock files for reproducible builds:

```bash
# Generate lock file
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

### Caching

UV automatically caches downloaded packages for faster re-installs:

```bash
# View cache location
uv cache dir

# Clear cache if needed
uv cache clean
```

### Parallel Downloads

UV downloads and installs packages in parallel automatically - no configuration needed!

## Learn More

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs Pip Benchmarks](https://github.com/astral-sh/uv#benchmarks)
- [Python Packaging with UV](https://docs.astral.sh/uv/)

## Migration from Pip

If you were using pip before:

1. **Install UV**: `make install-uv`
2. **Recreate venv**: `rm -rf .venv && uv venv --python 3.13`
3. **Install deps**: `uv pip install -e ".[dev]"`
4. **Use UV commands**: Replace `pip` with `uv pip` in your workflow

Everything else stays the same - UV is a drop-in replacement!
